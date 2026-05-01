#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <functional>
#include <chrono>

#include <raylib.h>
#include <raymath.h>

#include "global.h"
#include "sph2d.h"
#include "shape.h"
#include "spatial_hash.h"

using namespace sph2d;
using namespace sph2d::shape;
using namespace sph2d::collection;

// #define TEST

#if VERLET_LIST == 1
void computeDensityPressure(
    ParticleSystem& system, 
    const SpatialHashGridSoA& grid
) {
    int w = grid.grid_w;
    int h = grid.grid_h;
    int n = grid.grid_n;

    // ! var 1 basic density calculation
    const __m512 v_H2 = _mm512_set1_ps(H2);
    const __m512 v_poly6_factor = _mm512_set1_ps(alpha_poly6);

    // for each cell, process all its particles
    // beacause there is ghost cell, we start with grid_w + 1
    #pragma omp parallel for schedule(dynamic, DYNAMIC_SCHEDULE_CELL_BASED)
    for (int c = w + 1; c <= h*w-2; c++) {
        int i_start = grid.cell_start[c];
        int i_end = grid.cell_start[c+1];

        // [i_start, i_end)
        // for each particle in the cell
        for (int i = i_start; i < i_end; i++) {

            __m512 v_xi = _mm512_set1_ps(system.x[i]);
            __m512 v_yi = _mm512_set1_ps(system.y[i]);
            __m512 v_rho_acc = _mm512_setzero_ps();
            
            float rhoi = 0.0f;
            // for each other particle in the neighbouring cell (self-cell included)
            for (const int nc : grid.getNeighbourCells(c)) {
                int j_start = grid.cell_start[nc]; 
                int j_end = grid.cell_start[nc+1];
                int j_count = j_end - j_start;

                // [j_start, j_end)
                // vectorized 16 particles in a time
                for (int j = 0; j < j_count; j += 16) {
                    // i-th particle itself should be included
                    // so no if (i != j) { }
                    int remaining = j_count - j;
                    __mmask16 mask = (remaining >= 16) ? 0xFFFF : (1U << remaining) - 1;

                    int curr_j = j_start + j;
                    
#if SOFTWARE_PREFETCH == 2
                    if (curr_j + SOFTWARE_PREFETCH_DIST < j_end) {
                        _mm_prefetch((const char*)&system.x[curr_j + SOFTWARE_PREFETCH_DIST], _MM_HINT_T0);
                        _mm_prefetch((const char*)&system.y[curr_j + SOFTWARE_PREFETCH_DIST], _MM_HINT_T0);
                        _mm_prefetch((const char*)&system.vx[curr_j + SOFTWARE_PREFETCH_DIST], _MM_HINT_T0);
                        _mm_prefetch((const char*)&system.vy[curr_j + SOFTWARE_PREFETCH_DIST], _MM_HINT_T0);
                    }
#endif

                    __m512 v_xj = _mm512_maskz_loadu_ps(mask, &system.x[curr_j]);
                    __m512 v_yj = _mm512_maskz_loadu_ps(mask, &system.y[curr_j]);
                    __m512 v_mj = _mm512_maskz_loadu_ps(mask, &system.mass[curr_j]);

                    __m512 v_dx = _mm512_sub_ps(v_xi, v_xj);
                    __m512 v_dy = _mm512_sub_ps(v_yi, v_yj);
                    __m512 v_r2 = _mm512_mul_ps(v_dx, v_dx);
                    // fma
                    v_r2 = _mm512_fmadd_ps(v_dy, v_dy, v_r2);

                    // range mask, if r2 < H2
                    __mmask16 range_mask = _mm512_mask_cmp_ps_mask(mask, v_r2, v_H2, _CMP_LT_OQ);

                    if (range_mask > 0) {
                        __m512 v_w = get_W_poly6_simd(v_r2, v_H2, v_poly6_factor);
                        v_rho_acc = _mm512_mask3_fmadd_ps(v_mj, v_w, v_rho_acc, range_mask);
                    }
                }
            }
            system.rho[i] = _mm512_reduce_add_ps(v_rho_acc);
            system.pressure[i] = get_pressure(system.rho[i]);
        }
    }
}

void computeStress(ParticleSystem& system, const SpatialHashGridSoA& grid) {
    int w = grid.grid_w;
    int h = grid.grid_h;
    const __m512 v_H2 = _mm512_set1_ps(H2);
    const __m512 v_f_grad = _mm512_set1_ps(beta_poly6);

    #pragma omp parallel for schedule(dynamic, DYNAMIC_SCHEDULE_CELL_BASED)
    for (int c = w + 1; c <= h*w-2; c++) {
        int i_start = grid.cell_start[c];
        int i_end = grid.cell_start[c+1];

        // [i_start, i_end)
        // for each particle in the cell
        for (int i = i_start; i < i_end; i++) {
            
            // properties of particle i
            __m512 v_xi = _mm512_set1_ps(system.x[i]);
            __m512 v_yi = _mm512_set1_ps(system.y[i]);
            __m512 v_vxi = _mm512_set1_ps(system.vx[i]);
            __m512 v_vyi = _mm512_set1_ps(system.vy[i]);

            // accumulator for dv/dx, dv/dy, dW/dx, dW/dy
            __m512 v_dvx_dx = _mm512_setzero_ps();
            __m512 v_dvx_dy = _mm512_setzero_ps();
            __m512 v_dvy_dx = _mm512_setzero_ps();
            __m512 v_dvy_dy = _mm512_setzero_ps();

            for (const int nc : grid.getNeighbourCells(c)) {
                int j_start = grid.cell_start[nc]; 
                int j_end = grid.cell_start[nc+1];
                int j_count = j_end - j_start;

                for (int j = 0; j < j_count; j += 16) {
                    int remaining = j_count - j;
                    __mmask16 mask = (remaining >= 16) ? 0xFFFF : (1U << remaining) - 1;
                    int curr_j = j_start + j;


#if SOFTWARE_PREFETCH == 2
                    if (curr_j + SOFTWARE_PREFETCH_DIST < j_end) {
                        _mm_prefetch((const char*)&system.x[curr_j + SOFTWARE_PREFETCH_DIST], _MM_HINT_T0);
                        _mm_prefetch((const char*)&system.y[curr_j + SOFTWARE_PREFETCH_DIST], _MM_HINT_T0);
                        _mm_prefetch((const char*)&system.vx[curr_j + SOFTWARE_PREFETCH_DIST], _MM_HINT_T0);
                        _mm_prefetch((const char*)&system.vy[curr_j + SOFTWARE_PREFETCH_DIST], _MM_HINT_T0);
                    }
#endif

                    // properties of particle j
                    __m512 v_xj = _mm512_maskz_loadu_ps(mask, &system.x[curr_j]);
                    __m512 v_yj = _mm512_maskz_loadu_ps(mask, &system.y[curr_j]);
                    __m512 v_vxj = _mm512_maskz_loadu_ps(mask, &system.vx[curr_j]);
                    __m512 v_vyj = _mm512_maskz_loadu_ps(mask, &system.vy[curr_j]);
                    __m512 v_mj = _mm512_maskz_loadu_ps(mask, &system.mass[curr_j]);
                    __m512 v_rhoj = _mm512_maskz_loadu_ps(mask, &system.rho[curr_j]);

                    // calculate dx, dy, r2
                    __m512 v_dx = _mm512_sub_ps(v_xi, v_xj);
                    __m512 v_dy = _mm512_sub_ps(v_yi, v_yj);
                    __m512 v_r2 = _mm512_fmadd_ps(v_dx, v_dx, _mm512_mul_ps(v_dy, v_dy));

                    __mmask16 range_mask = _mm512_mask_cmp_ps_mask(mask, v_r2, v_H2, _CMP_LT_OQ);

                    if (range_mask > 0) {
                        // use reciprocal instruction to replace divide
                        // calculate V = m / rho

                        // ! var 1: use divide
#if RECIPROCAL_REPLACEMENT == 1
                        __m512 v_vol = _mm512_div_ps(v_mj, v_rhoj);
#else if RECIPROCAL_REPLACEMENT == 2
                        // ! var 2: use reciprocal
                        // __m512 v_inv_rhoj = _mm512_rcp14_ps(v_rhoj); 
                        // __m512 v_vol = _mm512_mul_ps(v_mj, v_inv_rhoj);

                        // ! var 3: or better, use Newton's method to refine the reciprocal
                        __m512 v_inv_rhoj = _mm512_rcp14_ps(v_rhoj); 
                        
                        __m512 v_two = _mm512_set1_ps(2.0f);
                        __m512 v_refine = _mm512_fnmadd_ps(v_rhoj, v_inv_rhoj, v_two); // (2 - rho * inv_rho)
                        v_inv_rhoj = _mm512_mul_ps(v_inv_rhoj, v_refine); 

                        __m512 v_vol = _mm512_mul_ps(v_mj, v_inv_rhoj);
#endif
                        
                        // calculate dv = vj - vi
                        __m512 v_dvx = _mm512_sub_ps(v_vxj, v_vxi);
                        __m512 v_dvy = _mm512_sub_ps(v_vyj, v_vyi);

                        // calculate dW/dx, dW/dy
                        __m512 v_dWx, v_dWy;
                        get_dW_dxi_poly6_simd(v_dx, v_dy, v_r2, v_H2, v_f_grad, v_dWx, v_dWy, range_mask);

                        // accumulate: dv_dx += V * dvx * dWx ...
                        __m512 v_val = _mm512_mul_ps(v_vol, v_dWx);
                        v_dvx_dx = _mm512_mask3_fmadd_ps(v_dvx, v_val, v_dvx_dx, range_mask);
                        v_dvy_dx = _mm512_mask3_fmadd_ps(v_dvy, v_val, v_dvy_dx, range_mask);

                        v_val = _mm512_mul_ps(v_vol, v_dWy);
                        v_dvx_dy = _mm512_mask3_fmadd_ps(v_dvx, v_val, v_dvx_dy, range_mask);
                        v_dvy_dy = _mm512_mask3_fmadd_ps(v_dvy, v_val, v_dvy_dy, range_mask);
                    }
                }
            }

            // restore scalar values
            float dvx_dx = _mm512_reduce_add_ps(v_dvx_dx);
            float dvx_dy = _mm512_reduce_add_ps(v_dvx_dy);
            float dvy_dx = _mm512_reduce_add_ps(v_dvy_dx);
            float dvy_dy = _mm512_reduce_add_ps(v_dvy_dy);
            float divv = dvx_dx + dvy_dy;
            
            // physical calculation
            if constexpr (IDEAL_FLUID == 1) {
                system.pxx[i] = -system.pressure[i];
                system.pxy[i] = 0.0f;
                system.pyy[i] = -system.pressure[i];
            } else {
                system.pxx[i] = -system.pressure[i] - _2_3_VISC * divv + _2VISC * dvx_dx;
                system.pxy[i] = VISC * (dvx_dy + dvy_dx);
                system.pyy[i] = -system.pressure[i] - _2_3_VISC * divv + _2VISC * dvy_dy;
            }
        }
}
}

// force
void computeAcceleration(ParticleSystem& system, const SpatialHashGridSoA& grid) {
    int w = grid.grid_w;
    int h = grid.grid_h;

    // pre-load all constant vectors, greatly reducing register pressure
    const __m512 v_H2 = _mm512_set1_ps(H2);
    const __m512 v_H = _mm512_set1_ps(H);
    const __m512 v_eps_H2 = _mm512_set1_ps(0.01f * H2);
    const __m512 v_f_grad = _mm512_set1_ps(beta_poly6);
    const __m512 v_half = _mm512_set1_ps(0.5f);
    const __m512 v_m_alpha_c = _mm512_set1_ps(-MONAGHAN_ALPHA * CS);
    const __m512 v_beta = _mm512_set1_ps(MONAGHAN_BETA);
    const __m512 v_ones = _mm512_set1_ps(1.0f);
    const __m512 v_zero = _mm512_setzero_ps();

    // skip the outer Ghost Cells
    #pragma omp parallel for schedule(dynamic, DYNAMIC_SCHEDULE_CELL_BASED)
    for (int c = w + 1; c <= h*w-2; c++) {
        int i_start = grid.cell_start[c];
        int i_end = grid.cell_start[c+1];
        
        // [i_start, i_end)
        // for each particle in the cell
        for (int i = i_start; i < i_end; i++) {
            
            // 1. pre-calculate properties of particle i
            float rhoi = system.rho[i];
            float inv_rhoi2_scalar = 1.0f / (rhoi * rhoi);
            
            __m512 v_xi = _mm512_set1_ps(system.x[i]);
            __m512 v_yi = _mm512_set1_ps(system.y[i]);
            __m512 v_vxi = _mm512_set1_ps(system.vx[i]);
            __m512 v_vyi = _mm512_set1_ps(system.vy[i]);
            __m512 v_rhoi = _mm512_set1_ps(rhoi);
            
            // pre-calculate p/rho^2, as a constant vector for accumulation
            __m512 v_pxxi_rhoi = _mm512_set1_ps(system.pxx[i] * inv_rhoi2_scalar);
            __m512 v_pxyi_rhoi = _mm512_set1_ps(system.pxy[i] * inv_rhoi2_scalar);
            __m512 v_pyyi_rhoi = _mm512_set1_ps(system.pyy[i] * inv_rhoi2_scalar);

            __m512 v_axi = _mm512_setzero_ps();
            __m512 v_ayi = _mm512_setzero_ps();

            for (const int nc : grid.getNeighbourCells(c)) {
                int j_start = grid.cell_start[nc]; 
                int j_end = grid.cell_start[nc+1];
                int j_count = j_end - j_start;

                for (int j = 0; j < j_count; j += 16) {
                    int remaining = j_count - j;
                    __mmask16 mask = (remaining >= 16) ? 0xFFFF : (1U << remaining) - 1;
                    int curr_j = j_start + j;

#if SOFTWARE_PREFETCH == 2
                    if (curr_j + SOFTWARE_PREFETCH_DIST < j_end) {
                        _mm_prefetch((const char*)&system.x[curr_j + SOFTWARE_PREFETCH_DIST], _MM_HINT_T0);
                        _mm_prefetch((const char*)&system.y[curr_j + SOFTWARE_PREFETCH_DIST], _MM_HINT_T0);
                        _mm_prefetch((const char*)&system.vx[curr_j + SOFTWARE_PREFETCH_DIST], _MM_HINT_T0);
                        _mm_prefetch((const char*)&system.vy[curr_j + SOFTWARE_PREFETCH_DIST], _MM_HINT_T0);
                    }
#endif

                    // step A (lazy loading optimization)
                    __m512 v_xj = _mm512_maskz_loadu_ps(mask, &system.x[curr_j]);
                    __m512 v_yj = _mm512_maskz_loadu_ps(mask, &system.y[curr_j]);
                    
                    __m512 v_dx = _mm512_sub_ps(v_xi, v_xj);
                    __m512 v_dy = _mm512_sub_ps(v_yi, v_yj);
                    __m512 v_r2 = _mm512_fmadd_ps(v_dx, v_dx, _mm512_mul_ps(v_dy, v_dy));

                    // particles within the smoothing radius
                    __mmask16 range_mask = _mm512_mask_cmp_ps_mask(mask, v_r2, v_H2, _CMP_LT_OQ);

                    if (range_mask > 0) {
                        __m512 v_vxj  = _mm512_maskz_loadu_ps(range_mask, &system.vx[curr_j]);
                        __m512 v_vyj  = _mm512_maskz_loadu_ps(range_mask, &system.vy[curr_j]);
                        __m512 v_mj   = _mm512_maskz_loadu_ps(range_mask, &system.mass[curr_j]);
                        __m512 v_rhoj = _mm512_maskz_loadu_ps(range_mask, &system.rho[curr_j]);
                        __m512 v_pxxj = _mm512_maskz_loadu_ps(range_mask, &system.pxx[curr_j]);
                        __m512 v_pxyj = _mm512_maskz_loadu_ps(range_mask, &system.pxy[curr_j]);
                        __m512 v_pyyj = _mm512_maskz_loadu_ps(range_mask, &system.pyy[curr_j]);

                        // calculate dW
                        __m512 v_dWx, v_dWy;
                        get_dW_dxi_poly6_simd(v_dx, v_dy, v_r2, v_H2, v_f_grad, v_dWx, v_dWy, range_mask);

                        // --- 1. calculate stress tensor ---
                        __m512 v_rhoj2 = _mm512_mul_ps(v_rhoj, v_rhoj);
                        __m512 v_inv_rhoj2 = _mm512_div_ps(v_ones, v_rhoj2);
                        
                        // term1, 2, 3 = (p_i/rho_i^2) + (p_j/rho_j^2)
                        __m512 v_term1 = _mm512_fmadd_ps(v_pxxj, v_inv_rhoj2, v_pxxi_rhoi);
                        __m512 v_term2 = _mm512_fmadd_ps(v_pxyj, v_inv_rhoj2, v_pxyi_rhoi);
                        __m512 v_term3 = _mm512_fmadd_ps(v_pyyj, v_inv_rhoj2, v_pyyi_rhoi);

                        // accumulate: axi += mj * (term1*dWx + term2*dWy)
                        __m512 v_ax_step = _mm512_fmadd_ps(v_term1, v_dWx, _mm512_mul_ps(v_term2, v_dWy));
                        v_axi = _mm512_mask3_fmadd_ps(v_mj, v_ax_step, v_axi, range_mask);

                        // accumulate: ayi += mj * (term2*dWx + term3*dWy)
                        __m512 v_ay_step = _mm512_fmadd_ps(v_term2, v_dWx, _mm512_mul_ps(v_term3, v_dWy));
                        v_ayi = _mm512_mask3_fmadd_ps(v_mj, v_ay_step, v_ayi, range_mask);

                        // --- 2. calculate artificial viscosity (Monaghan) ---
                        __m512 v_dvx = _mm512_sub_ps(v_vxi, v_vxj);
                        __m512 v_dvy = _mm512_sub_ps(v_vyi, v_vyj);
                        // vdotr = dvx*dx + dvy*dy
                        __m512 v_vdotr = _mm512_fmadd_ps(v_dvx, v_dx, _mm512_mul_ps(v_dvy, v_dy));

                        // only vdotr < 0 calculate viscosity
                        __mmask16 visc_mask = _mm512_mask_cmp_ps_mask(range_mask, v_vdotr, v_zero, _CMP_LT_OQ);

                        if (visc_mask > 0) {
                            __m512 v_denom = _mm512_add_ps(v_r2, v_eps_H2);
#if RECIPROCAL_REPLACEMENT == 1         // ! var 1: normal division            
                            __m512 v_mu = _mm512_mask_div_ps(v_zero, visc_mask, _mm512_mul_ps(v_H, v_vdotr), v_denom);

                            __m512 v_rhoAvg = _mm512_mul_ps(v_half, _mm512_add_ps(v_rhoi, v_rhoj));
                            
                            // pi_ij = (-alpha*CS*mu + beta*mu^2) / rhoAvg  => mu*(beta*mu - alpha*CS) / rhoAvg
                            __m512 v_pi = _mm512_fmadd_ps(v_beta, v_mu, v_m_alpha_c);
                            v_pi = _mm512_mul_ps(v_pi, v_mu);
                            v_pi = _mm512_mask_div_ps(v_zero, visc_mask, v_pi, v_rhoAvg);
                                                        
#else if RECIPROCAL_REPLACEMENT == 2    // ! var 2: use reciprocal instruction
                            // reciprocal
                            __m512 v_inv_denom = _mm512_maskz_rcp14_ps(visc_mask, v_denom); 
                            __m512 v_mu = _mm512_mul_ps(_mm512_mul_ps(v_H, v_vdotr), v_inv_denom);
                            
                            __m512 v_rhoAvg = _mm512_mul_ps(v_half, _mm512_add_ps(v_rhoi, v_rhoj));
                            __m512 v_inv_rhoAvg = _mm512_maskz_rcp14_ps(visc_mask, v_rhoAvg); 
                            
                            __m512 v_pi = _mm512_fmadd_ps(v_beta, v_mu, v_m_alpha_c);
                            v_pi = _mm512_mul_ps(v_pi, v_mu);
                            v_pi = _mm512_mul_ps(v_pi, v_inv_rhoAvg); 
                            
#endif

                            __m512 v_mj_pi = _mm512_mul_ps(v_mj, v_pi);
                            // axi -= mj * pi * dWx => -(mj_pi * dWx) + axi
                            // FNMADD: -(A*B) + C
                            v_axi = _mm512_mask3_fnmadd_ps(v_mj_pi, v_dWx, v_axi, visc_mask);
                            v_ayi = _mm512_mask3_fnmadd_ps(v_mj_pi, v_dWy, v_ayi, visc_mask);
                        }
                    }
                }
            }
            
            // reduce and write back to system
            system.ax[i] = _mm512_reduce_add_ps(v_axi);
            system.ay[i] = _mm512_reduce_add_ps(v_ayi) + GRAV;
        }
    }
}

#else if VERLET_LIST == 2
void computeDensityPressure(
    ParticleSystem& system, 
    const SpatialHashGridSoA& grid
) {
    int n = system.x.size();

    // 预加载常数
    const __m512 v_H2 = _mm512_set1_ps(H2);
    const __m512 v_poly6_factor = _mm512_set1_ps(alpha_poly6);

    // 外层循环大幅简化：直接遍历所有粒子即可，无需关心二维网格边界
    #pragma omp parallel for schedule(dynamic, DYNAMIC_SCHEDULE_PARTICLE_BASED)
    for (int i = 0; i < n; i++) {

        __m512 v_xi = _mm512_set1_ps(system.x[i]);
        __m512 v_yi = _mm512_set1_ps(system.y[i]);
        __m512 v_rho_acc = _mm512_setzero_ps();
        
        // 1. 获取当前粒子的邻居数量和在一维缓存数组中的起始位置
        int neighbor_count = grid.neighbour_count[i];
        int base_idx = i * SpatialHashGridSoA::MAX_NEIGHBOURS;

        // 2. 遍历该粒子的所有邻居（按 16 个一组进行向量化）
        for (int j = 0; j < neighbor_count; j += 16) {
            
            int remaining = neighbor_count - j;
            __mmask16 mask = (remaining >= 16) ? 0xFFFF : (1U << remaining) - 1;

            // ----------------------------------------------------
            // 核心改变：如何加载离散的邻居数据
            // ----------------------------------------------------
            
            // 步骤 A: 加载 16 个邻居的索引 ID (整型向量)
            __m512i v_j_indices = _mm512_maskz_loadu_epi32(
                mask, 
                &grid.neighbour_list_flat[base_idx + j]
            );

            // 软件预取：由于 Gather 指令比较慢，在这里预取下一批邻居的索引是非常好的选择
#if SOFTWARE_PREFETCH == 2
            if (j + SOFTWARE_PREFETCH_DIST < neighbor_count) {
                _mm_prefetch((const char*)&grid.neighbour_list_flat[base_idx + j + SOFTWARE_PREFETCH_DIST], _MM_HINT_T0);
            }
#endif

            // 步骤 B: 使用 _mm512_i32gather_ps 根据索引获取离散的浮点数据
            // 参数解释: (初始值, 掩码, 索引向量, 数组基地址, 缩放因子(float占用4字节))
            __m512 v_xj = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, v_j_indices, system.x.data(), 4);
            __m512 v_yj = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, v_j_indices, system.y.data(), 4);
            __m512 v_mj = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, v_j_indices, system.mass.data(), 4);

            // ----------------------------------------------------
            // 以下物理计算逻辑与之前完全相同
            // ----------------------------------------------------
            __m512 v_dx = _mm512_sub_ps(v_xi, v_xj);
            __m512 v_dy = _mm512_sub_ps(v_yi, v_yj);
            __m512 v_r2 = _mm512_mul_ps(v_dx, v_dx);
            v_r2 = _mm512_fmadd_ps(v_dy, v_dy, v_r2);

            // 二次距离判断：因为 Verlet List 的范围比 H 大，所以依然需要这个判断
            __mmask16 range_mask = _mm512_mask_cmp_ps_mask(mask, v_r2, v_H2, _CMP_LT_OQ);

            if (range_mask > 0) {
                __m512 v_w = get_W_poly6_simd(v_r2, v_H2, v_poly6_factor);
                v_rho_acc = _mm512_mask3_fmadd_ps(v_mj, v_w, v_rho_acc, range_mask);
            }
        }
        
        system.rho[i] = _mm512_reduce_add_ps(v_rho_acc);
        system.pressure[i] = get_pressure(system.rho[i]);
    }
}

void computeStress(ParticleSystem& system, const SpatialHashGridSoA& grid) {
    int n = system.x.size();
    const __m512 v_H2 = _mm512_set1_ps(H2);
    const __m512 v_f_grad = _mm512_set1_ps(beta_poly6);

    // 外层循环：直接并行遍历所有粒子
    #pragma omp parallel for schedule(dynamic, DYNAMIC_SCHEDULE_PARTICLE_BASED)
    for (int i = 0; i < n; i++) {
        
        // properties of particle i
        __m512 v_xi = _mm512_set1_ps(system.x[i]);
        __m512 v_yi = _mm512_set1_ps(system.y[i]);
        __m512 v_vxi = _mm512_set1_ps(system.vx[i]);
        __m512 v_vyi = _mm512_set1_ps(system.vy[i]);

        // accumulator for dv/dx, dv/dy, dW/dx, dW/dy
        __m512 v_dvx_dx = _mm512_setzero_ps();
        __m512 v_dvx_dy = _mm512_setzero_ps();
        __m512 v_dvy_dx = _mm512_setzero_ps();
        __m512 v_dvy_dy = _mm512_setzero_ps();

        // 1. 获取当前粒子在 Verlet List 中的邻居数量和数组基底偏移
        int neighbor_count = grid.neighbour_count[i];
        int base_idx = i * SpatialHashGridSoA::MAX_NEIGHBOURS;
        
        // 2. 每次处理 16 个邻居
        for (int j = 0; j < neighbor_count; j += 16) {
            int remaining = neighbor_count - j;
            __mmask16 mask = (remaining >= 16) ? 0xFFFF : (1U << remaining) - 1;

            // --- 核心修改：利用掩码加载 16 个邻居的整数 ID 索引 ---
            __m512i v_j_indices = _mm512_maskz_loadu_epi32(
                mask, 
                &grid.neighbour_list_flat[base_idx + j]
            );

#if SOFTWARE_PREFETCH == 2
            // 预取下一批邻居的索引 ID。由于 Gather 指令需要依据索引去抓取，
            // 保证索引数组始终在缓存中是提升后续 Gather 效率的关键。
            if (j + SOFTWARE_PREFETCH_DIST < neighbor_count) {
                _mm_prefetch((const char*)&grid.neighbour_list_flat[base_idx + j + SOFTWARE_PREFETCH_DIST], _MM_HINT_T0);
            }
#endif

            // --- 核心修改：使用 Gather 指令加载离散属性 ---
            // properties of particle j
            __m512 v_xj = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, v_j_indices, system.x.data(), 4);
            __m512 v_yj = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, v_j_indices, system.y.data(), 4);
            __m512 v_vxj = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, v_j_indices, system.vx.data(), 4);
            __m512 v_vyj = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, v_j_indices, system.vy.data(), 4);
            __m512 v_mj = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, v_j_indices, system.mass.data(), 4);
            __m512 v_rhoj = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, v_j_indices, system.rho.data(), 4);

            // calculate dx, dy, r2
            __m512 v_dx = _mm512_sub_ps(v_xi, v_xj);
            __m512 v_dy = _mm512_sub_ps(v_yi, v_yj);
            __m512 v_r2 = _mm512_fmadd_ps(v_dx, v_dx, _mm512_mul_ps(v_dy, v_dy));

            // 二次距离判断：因为 Verlet 搜索半径更宽，这里必须进行严谨的 H2 距离筛选
            __mmask16 range_mask = _mm512_mask_cmp_ps_mask(mask, v_r2, v_H2, _CMP_LT_OQ);

            if (range_mask > 0) {
                // calculate V = m / rho
                // 完美保留了你的高级倒数优化逻辑

                // ! var 1: use divide
#if RECIPROCAL_REPLACEMENT == 1
                __m512 v_vol = _mm512_maskz_div_ps(range_mask, v_mj, v_rhoj);
#elif RECIPROCAL_REPLACEMENT == 2
                // ! var 2: use reciprocal
                // __m512 v_inv_rhoj = _mm512_maskz_rcp14_ps(range_mask, v_rhoj); 
                // __m512 v_vol = _mm512_mul_ps(v_mj, v_inv_rhoj);

                // ! var 3: or better, use Newton's method to refine the reciprocal
                __m512 v_inv_rhoj = _mm512_maskz_rcp14_ps(range_mask, v_rhoj); 
                
                __m512 v_two = _mm512_set1_ps(2.0f);
                __m512 v_refine = _mm512_fnmadd_ps(v_rhoj, v_inv_rhoj, v_two); // (2 - rho * inv_rho)
                v_inv_rhoj = _mm512_mul_ps(v_inv_rhoj, v_refine); 

                __m512 v_vol = _mm512_mul_ps(v_mj, v_inv_rhoj);
#endif
                
                // calculate dv = vj - vi
                __m512 v_dvx = _mm512_sub_ps(v_vxj, v_vxi);
                __m512 v_dvy = _mm512_sub_ps(v_vyj, v_vyi);

                // calculate dW/dx, dW/dy
                __m512 v_dWx, v_dWy;
                get_dW_dxi_poly6_simd(v_dx, v_dy, v_r2, v_H2, v_f_grad, v_dWx, v_dWy, range_mask);

                // accumulate: dv_dx += V * dvx * dWx ...
                __m512 v_val = _mm512_mul_ps(v_vol, v_dWx);
                v_dvx_dx = _mm512_mask3_fmadd_ps(v_dvx, v_val, v_dvx_dx, range_mask);
                v_dvy_dx = _mm512_mask3_fmadd_ps(v_dvy, v_val, v_dvy_dx, range_mask);

                v_val = _mm512_mul_ps(v_vol, v_dWy);
                v_dvx_dy = _mm512_mask3_fmadd_ps(v_dvx, v_val, v_dvx_dy, range_mask);
                v_dvy_dy = _mm512_mask3_fmadd_ps(v_dvy, v_val, v_dvy_dy, range_mask);
            }
        }

        // restore scalar values
        float dvx_dx = _mm512_reduce_add_ps(v_dvx_dx);
        float dvx_dy = _mm512_reduce_add_ps(v_dvx_dy);
        float dvy_dx = _mm512_reduce_add_ps(v_dvy_dx);
        float dvy_dy = _mm512_reduce_add_ps(v_dvy_dy);
        float divv = dvx_dx + dvy_dy;
        
        // physical calculation
        if constexpr (IDEAL_FLUID == 1) {
            system.pxx[i] = -system.pressure[i];
            system.pxy[i] = 0.0f;
            system.pyy[i] = -system.pressure[i];
        } else {
            system.pxx[i] = -system.pressure[i] - _2_3_VISC * divv + _2VISC * dvx_dx;
            system.pxy[i] = VISC * (dvx_dy + dvy_dx);
            system.pyy[i] = -system.pressure[i] - _2_3_VISC * divv + _2VISC * dvy_dy;
        }
    }
}

// acceleration
void computeAcceleration(ParticleSystem& system, const SpatialHashGridSoA& grid) {
    int n = system.x.size();

    // 预加载所有不变的常数向量，极大地节省寄存器压力
    const __m512 v_H2 = _mm512_set1_ps(H2);
    const __m512 v_H = _mm512_set1_ps(H);
    const __m512 v_eps_H2 = _mm512_set1_ps(0.01f * H2);
    const __m512 v_f_grad = _mm512_set1_ps(beta_poly6);
    const __m512 v_half = _mm512_set1_ps(0.5f);
    const __m512 v_m_alpha_c = _mm512_set1_ps(-MONAGHAN_ALPHA * CS);
    const __m512 v_beta = _mm512_set1_ps(MONAGHAN_BETA);
    const __m512 v_ones = _mm512_set1_ps(1.0f);
    const __m512 v_zero = _mm512_setzero_ps();

    // 彻底消灭二维网格循环，直接一维展开遍历所有粒子
    #pragma omp parallel for schedule(dynamic, DYNAMIC_SCHEDULE_PARTICLE_BASED)
    for (int i = 0; i < n; i++) {
        
        // 1. 预计算中心粒子 i 的标量属性和向量属性
        float rhoi = system.rho[i];
        float inv_rhoi2_scalar = 1.0f / (rhoi * rhoi);
        
        __m512 v_xi = _mm512_set1_ps(system.x[i]);
        __m512 v_yi = _mm512_set1_ps(system.y[i]);
        __m512 v_vxi = _mm512_set1_ps(system.vx[i]);
        __m512 v_vyi = _mm512_set1_ps(system.vy[i]);
        __m512 v_rhoi = _mm512_set1_ps(rhoi);
        
        __m512 v_pxxi_rhoi = _mm512_set1_ps(system.pxx[i] * inv_rhoi2_scalar);
        __m512 v_pxyi_rhoi = _mm512_set1_ps(system.pxy[i] * inv_rhoi2_scalar);
        __m512 v_pyyi_rhoi = _mm512_set1_ps(system.pyy[i] * inv_rhoi2_scalar);

        __m512 v_axi = _mm512_setzero_ps();
        __m512 v_ayi = _mm512_setzero_ps();

        // 获取粒子 i 在邻居表中的数量和基底偏移
        int neighbor_count = grid.neighbour_count[i];
        int base_idx = i * SpatialHashGridSoA::MAX_NEIGHBOURS;

        // 每次并行处理 16 个离散的邻居粒子
        for (int j = 0; j < neighbor_count; j += 16) {
            int remaining = neighbor_count - j;
            __mmask16 mask = (remaining >= 16) ? 0xFFFF : (1U << remaining) - 1;

            // 加载当前这批邻居的整型 ID 索引
            __m512i v_j_indices = _mm512_maskz_loadu_epi32(
                mask, 
                &grid.neighbour_list_flat[base_idx + j]
            );

#if SOFTWARE_PREFETCH == 2
            // 预取下一批邻居的索引 ID。由于 Gather 严重依赖索引，
            // 提前将索引拉入 L1 缓存能最大程度掩盖访存延迟
            if (j + SOFTWARE_PREFETCH_DIST < neighbor_count) {
                _mm_prefetch((const char*)&grid.neighbour_list_flat[base_idx + j + SOFTWARE_PREFETCH_DIST], _MM_HINT_T0);
            }
#endif

            // 步骤 A：仅 Gather 坐标并验证距离 (保留了你的延迟加载优化逻辑)
            __m512 v_xj = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, v_j_indices, system.x.data(), 4);
            __m512 v_yj = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), mask, v_j_indices, system.y.data(), 4);
            
            __m512 v_dx = _mm512_sub_ps(v_xi, v_xj);
            __m512 v_dy = _mm512_sub_ps(v_yi, v_yj);
            __m512 v_r2 = _mm512_fmadd_ps(v_dx, v_dx, _mm512_mul_ps(v_dy, v_dy));

            // 二次过滤掩码: 因为 Verlet 列表边界更宽，必须重新确认是否处于平滑半径内
            __mmask16 range_mask = _mm512_mask_cmp_ps_mask(mask, v_r2, v_H2, _CMP_LT_OQ);

            if (range_mask > 0) {
                // 步骤 B：使用 range_mask 仅对真正需要的粒子执行重量级的 Gather！
                // 这将避免大量无用的内存抓取操作
                __m512 v_vxj  = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), range_mask, v_j_indices, system.vx.data(), 4);
                __m512 v_vyj  = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), range_mask, v_j_indices, system.vy.data(), 4);
                __m512 v_mj   = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), range_mask, v_j_indices, system.mass.data(), 4);
                __m512 v_rhoj = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), range_mask, v_j_indices, system.rho.data(), 4);
                __m512 v_pxxj = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), range_mask, v_j_indices, system.pxx.data(), 4);
                __m512 v_pxyj = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), range_mask, v_j_indices, system.pxy.data(), 4);
                __m512 v_pyyj = _mm512_mask_i32gather_ps(_mm512_setzero_ps(), range_mask, v_j_indices, system.pyy.data(), 4);

                // 计算梯度 dW
                __m512 v_dWx, v_dWy;
                get_dW_dxi_poly6_simd(v_dx, v_dy, v_r2, v_H2, v_f_grad, v_dWx, v_dWy, range_mask);

                // --- 1. 应力张量力计算 ---
                __m512 v_rhoj2 = _mm512_mul_ps(v_rhoj, v_rhoj);
                __m512 v_inv_rhoj2 = _mm512_div_ps(v_ones, v_rhoj2);
                
                __m512 v_term1 = _mm512_fmadd_ps(v_pxxj, v_inv_rhoj2, v_pxxi_rhoi);
                __m512 v_term2 = _mm512_fmadd_ps(v_pxyj, v_inv_rhoj2, v_pxyi_rhoi);
                __m512 v_term3 = _mm512_fmadd_ps(v_pyyj, v_inv_rhoj2, v_pyyi_rhoi);

                __m512 v_ax_step = _mm512_fmadd_ps(v_term1, v_dWx, _mm512_mul_ps(v_term2, v_dWy));
                v_axi = _mm512_mask3_fmadd_ps(v_mj, v_ax_step, v_axi, range_mask);

                __m512 v_ay_step = _mm512_fmadd_ps(v_term2, v_dWx, _mm512_mul_ps(v_term3, v_dWy));
                v_ayi = _mm512_mask3_fmadd_ps(v_mj, v_ay_step, v_ayi, range_mask);

                // --- 2. 人工黏性 (Monaghan) 计算 ---
                __m512 v_dvx = _mm512_sub_ps(v_vxi, v_vxj);
                __m512 v_dvy = _mm512_sub_ps(v_vyi, v_vyj);
                __m512 v_vdotr = _mm512_fmadd_ps(v_dvx, v_dx, _mm512_mul_ps(v_dvy, v_dy));

                // 三次过滤掩码：仅当相互靠近时 (vdotr < 0) 计算黏性
                __mmask16 visc_mask = _mm512_mask_cmp_ps_mask(range_mask, v_vdotr, v_zero, _CMP_LT_OQ);

                if (visc_mask > 0) {
                    __m512 v_denom = _mm512_add_ps(v_r2, v_eps_H2);
                    __m512 v_rhoAvg = _mm512_mul_ps(v_half, _mm512_add_ps(v_rhoi, v_rhoj));

#if RECIPROCAL_REPLACEMENT == 1         // ! var 1: normal division            
                    __m512 v_mu = _mm512_mask_div_ps(v_zero, visc_mask, _mm512_mul_ps(v_H, v_vdotr), v_denom);
                    
                    __m512 v_pi = _mm512_fmadd_ps(v_beta, v_mu, v_m_alpha_c);
                    v_pi = _mm512_mul_ps(v_pi, v_mu);
                    v_pi = _mm512_mask_div_ps(v_zero, visc_mask, v_pi, v_rhoAvg);
                                                            
#elif RECIPROCAL_REPLACEMENT == 2       // ! var 2: use reciprocal instruction
                    __m512 v_inv_denom = _mm512_maskz_rcp14_ps(visc_mask, v_denom); 
                    __m512 v_mu = _mm512_mul_ps(_mm512_mul_ps(v_H, v_vdotr), v_inv_denom);
                    
                    __m512 v_inv_rhoAvg = _mm512_maskz_rcp14_ps(visc_mask, v_rhoAvg); 
                    
                    __m512 v_pi = _mm512_fmadd_ps(v_beta, v_mu, v_m_alpha_c);
                    v_pi = _mm512_mul_ps(v_pi, v_mu);
                    v_pi = _mm512_mul_ps(v_pi, v_inv_rhoAvg); 
#endif
                    
                    __m512 v_mj_pi = _mm512_mul_ps(v_mj, v_pi);
                    
                    // FNMADD: -(A*B) + C
                    v_axi = _mm512_mask3_fnmadd_ps(v_mj_pi, v_dWx, v_axi, visc_mask);
                    v_ayi = _mm512_mask3_fnmadd_ps(v_mj_pi, v_dWy, v_ayi, visc_mask);
                }
            }
        }
        
        // 水平归约，添加外力并写回
        system.ax[i] = _mm512_reduce_add_ps(v_axi);
        system.ay[i] = _mm512_reduce_add_ps(v_ayi) + GRAV;
    }
}

#endif

#if BOUNDARY_PROCESS == 1
void integrate(ParticleSystem& system) {
    int n = system.x.size();

    #pragma omp parallel for schedule(dynamic, DYNAMIC_SCHEDULE_PARTICLE_BASED)
    for (int i = 0; i < n; i++) {
        system.vx[i]  += system.ax[i] * DT;
        system.vy[i]  += system.ay[i] * DT;

        system.x[i]  += system.vx[i] * DT;
        system.y[i]  += system.vy[i] * DT;

        if (system.x[i] < 0.0f) { 
            system.x[i] = 0.0f + BOUNDARY_SHIFT_EPSILON;        
            system.vx[i] *= DAMPING; 
        }
        else if (system.x[i] > physicalWidth) {
            system.x[i] = physicalWidth - BOUNDARY_SHIFT_EPSILON;
            system.vx[i] *= DAMPING; 
        }
        if (system.y[i] < 0.0f) {
            system.y[i] = 0.0f + BOUNDARY_SHIFT_EPSILON;
            system.vy[i] *= DAMPING; 
        }
        else if (system.y[i] > physicalHeight) { 
            system.y[i] = physicalHeight - BOUNDARY_SHIFT_EPSILON;
            system.vy[i] *= DAMPING; 
        }
    }
}
#else if BOUNDARY_PROCESS == 2

void integrate(ParticleSystem& system) {
    int n = system.x.size();

    #pragma omp parallel for schedule(dynamic, DYNAMIC_SCHEDULE_PARTICLE_BASED)
    for (int i = 0; i < n; i++) {
        float xi  = system.x[i];
        float yi  = system.y[i];
        float vxi = system.vx[i];
        float vyi = system.vy[i];
        
        float force_x = 0.0f;
        float force_y = 0.0f;

        // nonlinear repulsion force
        auto calculate_repulsion = [&](float dist, float vel) -> float {
            if (dist < PARTICLE_COLLISION_RADIUS) {
                float depth = PARTICLE_COLLISION_RADIUS - dist;
                return BOUNDARY_STIFFNESS * (depth * depth) - BOUNDARY_DAMPING * vel;
            }
            return 0.0f;
        };

        force_x += calculate_repulsion(xi, vxi);                                  
        force_x -= calculate_repulsion(physicalWidth - xi, -vxi);                 

        force_y += calculate_repulsion(yi, vyi);                                  
        force_y -= calculate_repulsion(physicalHeight - yi, -vyi);                

        // calculate total acceleration from pressure and repulsion force
        float total_ax = system.ax[i] + force_x;
        float total_ay = system.ay[i] + force_y;

        // update velocity
        vxi += total_ax * DT;
        vyi += total_ay * DT;

        // update position
        xi += vxi * DT;
        yi += vyi * DT;

        if (xi < 0.0f) { xi = 0.0f; vxi *= HARD_DAMPING; }
        else if (xi > physicalWidth) { xi = physicalWidth; vxi *= HARD_DAMPING; }
        
        if (yi < 0.0f) { yi = 0.0f; vyi *= HARD_DAMPING; }
        else if (yi > physicalHeight) { yi = physicalHeight; vyi *= HARD_DAMPING; }

        system.x[i]  = xi;
        system.y[i]  = yi;
        system.vx[i] = vxi;
        system.vy[i] = vyi;
    }
}

#endif

int main() {

    std::cout << "simulation step time: DT " << DT << "\n";

    InitWindow(screenWidth, screenHeight, "SPH 2D Fluid Insight");

    // init particles
    std::vector<Particle> particles = getParticles();
    ParticleSystem system(particles);

    int n = particles.size();
    particles.clear(); particles.shrink_to_fit();

    std::cout << "Total particles: " << particles.size() << std::endl;

    // arrange those paritcles in spatial hash struct
    SpatialHashGridSoA grid(physicalWidth, physicalHeight, H);

    Camera2D camera = { 0 };
    camera.zoom = 1.0f;

    SetTargetFPS(60);

    float simulatedTime = 0.0f;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    int iteration = 0;

#if VERLET_LIST == 2
    // for verlet list
    std::vector<float> last_build_x(system.x.size());
    std::vector<float> last_build_y(system.y.size());
    
    float max_displacement_sq = 0.0f;
    bool force_rebuild = true;          // rebuild verlet list or not
#endif

    while (!WindowShouldClose()) {
        // `substep` times iteration in one frame
#if VERLET_LIST == 1
        grid.build(system);
#endif 
        for(int i=0; i < substep; i++) {

#if VERLET_LIST == 2
            // ! when
            // ! 1. force_rebuild == true, 
            // ! 2. the max displacement of particles exceed the skin radius
            // ! rebuild spatial hash grid and verlet list
            if (4.0f * max_displacement_sq >= (VERLET_SKIN_RADIUS * VERLET_SKIN_RADIUS) || force_rebuild) {
                // rebuild spatial hash grid
                grid.build(system);
                // build verlet list based on spatial hash grid
                grid.buildVerletList(system, VERLET_SEARCH_RADIUS);

                #pragma omp parallel for schedule(static)
                for (int k = 0; k < n; k++) {
                    last_build_x[k] = system.x[k];
                    last_build_y[k] = system.y[k];
                }

                max_displacement_sq = 0.0f;
                force_rebuild = false;

                // std::cout << "Iteration: " << iteration << " rebuild verlet list.\n";
            }            
#endif

            computeDensityPressure(system, grid);
            computeStress(system, grid);
            computeAcceleration(system, grid);

            integrate(system);

#if VERLET_LIST == 2
            // trace the max displacement
            float current_max_disp_sq = 0.0f;
            #pragma omp parallel for schedule(static)
            for (int k = 0; k < n; k++) {
                float dx = system.x[k] - last_build_x[k];
                float dy = system.y[k] - last_build_y[k];
                float disp_sq = dx*dx + dy*dy;
                if (disp_sq > current_max_disp_sq) {
                    current_max_disp_sq = disp_sq;
                }
            }
            max_displacement_sq = current_max_disp_sq;
#endif

            simulatedTime += DT;
        }

        // print the real time cost in milliseconds for `ITERATION_TO_COUNT` iterations
        if (iteration % ITERATION_TO_COUNT == 0) {
            auto endTime = std::chrono::high_resolution_clock::now();
            std::cout << "iteration: " << iteration << ", time: " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << "ms\n";
        }

        float realTime = GetTime();

        BeginDrawing();
        ClearBackground(BLACK);
        BeginMode2D(camera);

            for (int i = 0; i < system.x.size(); i++) {
                float colorFactor = std::clamp(system.pressure[i] / 500.0f, 0.0f, 1.0f);
                Color c = {(unsigned char)(255 * colorFactor), (unsigned char)(100 + 155 * (1-colorFactor)), 255, 200};
                // Color c = {(unsigned char)255, (unsigned char)0, (unsigned char)0, (unsigned char)255};
                // map the physical position to screen position
                DrawCircleV(Vector2{PHYSICAL_SCREEN_MAPPING * system.x[i], PHYSICAL_SCREEN_MAPPING * system.y[i]}, PARTICLE_RADIUS, c); 
            }

        EndMode2D();
        DrawFPS(10, 10);
        DrawText(TextFormat("Particles: %d", system.x.size()), 10, 35, 20, RAYWHITE);      
        
        // simulated time and real time
        const char* simTimeText = TextFormat("Sim Time: %.2fs", simulatedTime);
        const char* realTimeText = TextFormat("Real Time: %.2fs", realTime);
        
        int fontSize = 20;
        int simTextWidth = MeasureText(simTimeText, fontSize);
        int realTextWidth = MeasureText(realTimeText, fontSize);

        DrawText(simTimeText, screenWidth - simTextWidth - 10, 10, fontSize, GREEN);
        DrawText(realTimeText, screenWidth - realTextWidth - 10, 35, fontSize, GREEN);

        EndDrawing();

        iteration++;
    }
    CloseWindow();
    return 0;
}