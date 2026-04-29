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
    int n = system.x.size();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {

if constexpr (RHO_VAR == 1) { // ! var 1 basic 

        float rhoi = 0.0f;
        grid.forEachNeighbour(i, system, H, H2, [&](int j) {
            float dx = system.x[i] - system.x[j];
            float dy = system.y[i] - system.y[j];
            float r = std::sqrt(dx*dx + dy*dy);
            rhoi += system.mass[j] * get_W(r);
        });
        system.rho[i] = rhoi;

} else if constexpr (RHO_VAR == 2) { // ! var 2 Shepard normalized

        float mass_total = 0.0f;
        float correction = 0.0f;

        grid.forEachNeighbour(i, system, H, H2, [&](int j) {
            float dx = system.x[i] - system.x[j];
            float dy = system.y[i] - system.y[j];

            float r = std::sqrt(dx*dx + dy*dy);
            float w = get_W(r);

            float tmp = system.mass[j] * w;
            mass_total += tmp;
            correction += tmp / system.rho[j]; 
        });

        system.rho[i] = mass_total / (correction + EPSILON);

} // var 2 end

        system.pressure[i] = get_pressure(system.rho[i]);
    }
}

void computeStress(    
    ParticleSystem& system, 
    const SpatialHashGridSoA& grid
) {
    int n = system.x.size();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float xi = system.x[i];
        float yi = system.y[i];
        
        float vxi = system.vx[i];
        float vyi = system.vy[i];

        // div(_v_)
        float divv = 0.0f;

        // dv/dx
        float dvx_dx = 0.0f;
        float dvx_dy = 0.0f;
        float dvy_dx = 0.0f;
        float dvy_dy = 0.0f;
        
        // sum them up
        grid.forEachNeighbour(i, system, H, H2, [&](int j) {
            float dx = xi - system.x[j];
            float dy = yi - system.y[j];

            float dvxji = system.vx[j] - vxi;
            float dvyji = system.vy[j] - vyi;

            float volume = system.mass[j] / system.rho[j];

            auto [dW_dx, dW_dy] = get_dW_dxi(dx, dy);

            // divv
            divv += volume * (dvxji * dW_dx + dvyji * dW_dy);

            // dv/dx
            dvx_dx += volume * dvxji * dW_dx;
            dvx_dy += volume * dvxji * dW_dy;
            dvy_dx += volume * dvyji * dW_dx;
            dvy_dy += volume * dvyji * dW_dy;
        });

if constexpr (IDEAL_FLUID == 1) {
        system.pxx[i] = -system.pressure[i];
        system.pxy[i] = 0.0f;
        system.pyy[i] = -system.pressure[i];
} else if constexpr (IDEAL_FLUID == 2) {
        system.pxx[i] = -system.pressure[i] - _2_3_VISC * divv + _2VISC * dvx_dx;
        system.pxy[i] = VISC * (dvx_dy + dvy_dx);
        system.pyy[i] = -system.pressure[i] - _2_3_VISC * divv + _2VISC * dvy_dy;
}

    }
}

// force
void computeAcceleration(    
    ParticleSystem& system,     
    const SpatialHashGridSoA& grid
) {
    int n = system.x.size();
    
    #pragma omp parallel for schedule(static)
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

            auto [dW_dx, dW_dy] = get_dW_dxi(dx, dy);

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
                float r = std::sqrt(dx*dx + dy*dy + 0.0001f * H2);
                float mu = H * vdotr / (r * r + 0.0001f * H2);
                float rhoAvg = 0.5f * (system.rho[i] + system.rho[j]);
                // alpha ~ 0.1, beta ~ 0.2, cs = speed of sound
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
    #pragma omp parallel for schedule(static)
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