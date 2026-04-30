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
    for (int y = 1; y < h - 1; y++) {
        for (int x = 1; x < w - 1; x++) {
            int c = y * w + x;
            for (int i = grid.cell_start[c]; i < grid.cell_start[c+1]; i++) {
                
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
}

// force
void computeAcceleration(    
    ParticleSystem& system,     
    const SpatialHashGridSoA& grid
) {
    int n = system.x.size();
    
    #pragma omp parallel for schedule(dynamic, DYNAMIC_SCHEDULE_CELL_BASED)
    for (int i = 0; i < n; i++) {
        float xi = system.x[i];
        float yi = system.y[i];

        float vxi = system.vx[i];
        float vyi = system.vy[i];
        
        float rhoi2 = system.rho[i] * system.rho[i];

        float pxxi = system.pxx[i];
        float pxyi = system.pxy[i];
        float pyyi = system.pyy[i];

        // acceleration
        float axi = 0.0f;
        float ayi = GRAV;

        grid.forEachNeighbour(i, system, H, H2, [&](int j) {
            float dx = xi - system.x[j];
            float dy = yi - system.y[j];

            float mj = system.mass[j];
            float rhoj2 = system.rho[j] * system.rho[j];
            
            float pxxj = system.pxx[j];
            float pxyj = system.pxy[j];
            float pyyj = system.pyy[j];

            auto [dW_dx, dW_dy] = get_dW_dxi_poly6(dx, dy);

            float term1 = (pxxi / rhoi2 + pxxj / rhoj2);
            float term2 = (pxyi / rhoi2 + pxyj / rhoj2);
            float term3 = (pyyi / rhoi2 + pyyj / rhoj2);
            
            axi += mj * (term1 * dW_dx + term2 * dW_dy);
            ayi += mj * (term2 * dW_dx + term3 * dW_dy);

            // monaghan artificial viscosity, avoiding from particles cross each other
            float dvx = system.vx[i] - system.vx[j];
            float dvy = system.vy[i] - system.vy[j];
            float vdotr = dvx * dx + dvy * dy;   // v_ij · r_ij

            if (vdotr < 0.0f) {
                float r2 = dx*dx + dy*dy;
                float denominator = r2 + 0.01f * H2; 
                float mu = H * vdotr / denominator;
                
                float rhoAvg = 0.5f * (system.rho[i] + system.rho[j]);
                float pi_ij = (-MONAGHAN_ALPHA * CS * mu + MONAGHAN_BETA * mu * mu) / rhoAvg;

                axi -= mj * pi_ij * dW_dx;
                ayi -= mj * pi_ij * dW_dy;
            }
        });

        system.ax[i] = axi;
        system.ay[i] = ayi;
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