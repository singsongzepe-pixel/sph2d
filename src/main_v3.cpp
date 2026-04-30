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
                        // calculate V = m / rho
                        __m512 v_vol = _mm512_div_ps(v_mj, v_rhoj);
                        
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
                            // mu = H * vdotr / denom (using masked division)
                            __m512 v_mu = _mm512_mask_div_ps(v_zero, visc_mask, _mm512_mul_ps(v_H, v_vdotr), v_denom);
                            
                            __m512 v_rhoAvg = _mm512_mul_ps(v_half, _mm512_add_ps(v_rhoi, v_rhoj));
                            
                            // pi_ij = (-alpha*CS*mu + beta*mu^2) / rhoAvg  => mu*(beta*mu - alpha*CS) / rhoAvg
                            __m512 v_pi = _mm512_fmadd_ps(v_beta, v_mu, v_m_alpha_c);
                            v_pi = _mm512_mul_ps(v_pi, v_mu);
                            v_pi = _mm512_mask_div_ps(v_zero, visc_mask, v_pi, v_rhoAvg);
                            
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

const float DAMPING = -0.5f; 
void integrate(ParticleSystem& system) {
    int n = system.x.size();

    #pragma omp parallel for schedule(dynamic, DYNAMIC_SCHEDULE_PARTICLE_BASED)
    for (int i = 0; i < n; i++) {
        system.vx[i]  += system.ax[i] * DT;
        system.vy[i]  += system.ay[i] * DT;

        system.x[i]  += system.vx[i] * DT;
        system.y[i]  += system.vy[i] * DT;

        if (system.x[i] < 0.0f) { 
            system.x[i] = 0.0f;        
            system.vx[i] *= DAMPING; 
        }
        else if (system.x[i] > physicalWidth) {
            system.x[i] = physicalWidth;
            system.vx[i] *= DAMPING; 
        }
        if (system.y[i] < 0.0f) {
            system.y[i] = 0.0f;
            system.vy[i] *= DAMPING; 
        }
        else if (system.y[i] > physicalHeight) { 
            system.y[i] = physicalHeight;
            system.vy[i] *= DAMPING; 
        }
    }
}


int main() {

    std::cout << "simulation step time: DT " << DT << "\n";

    InitWindow(screenWidth, screenHeight, "SPH 2D Fluid Insight");

    // init particles
    std::vector<Particle> particles = getParticles();
    ParticleSystem system(particles);
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

    while (!WindowShouldClose()) {
        // `substep` times iteration in one frame
        for(int i=0; i < substep; i++) {
            grid.build(system);
            
            computeDensityPressure(system, grid);
            computeStress(system, grid);
            computeAcceleration(system, grid);

            integrate(system);

            simulatedTime += DT;
        }

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