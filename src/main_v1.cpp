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
    std::vector<Particle>& particles, 
    const SpatialHashGrid& grid
) {
    int n = particles.size();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {

if constexpr (RHO_VAR == 1) { // ! var 1 basic 

        float rhoi = 0.0f;
        grid.forEachNeighbour(i, particles, H, H2, [&](int j) {
            float dx = particles[i].x - particles[j].x;
            float dy = particles[i].y - particles[j].y;
            float r = std::sqrt(dx*dx + dy*dy);
            rhoi += particles[j].mass * get_W(r);
        });
        particles[i].rho = rhoi;

} else if constexpr (RHO_VAR == 2) { // ! var 2 Shepard normalized

        float mass_total = 0.0f;
        float correction = 0.0f;

        grid.forEachNeighbour(i, particles, H, H2, [&](int j) {
            float dx = particles[i].x - particles[j].x;
            float dy = particles[i].y - particles[j].y;

            float r = std::sqrt(dx*dx + dy*dy);
            float w = get_W(r);

            float tmp = particles[j].mass * w;
            mass_total += tmp;
            correction += tmp / particles[j].rho; 
        });

        particles[i].rho = mass_total / (correction + EPSILON);

} // var 2 end

        particles[i].pressure = get_pressure(particles[i].rho);
    }
}

void computeStress(    
    std::vector<Particle>& particles, 
    const SpatialHashGrid& grid
) {
    int n = particles.size();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float xi = particles[i].x;
        float yi = particles[i].y;

        float vxi = particles[i].vx;
        float vyi = particles[i].vy;

        // div(_v_)
        float divv = 0.0f;

        // dv/dx
        float dvx_dx = 0.0f;
        float dvx_dy = 0.0f;
        float dvy_dx = 0.0f;
        float dvy_dy = 0.0f;
        
        // sum them up
        grid.forEachNeighbour(i, particles, H, H2, [&](int j) {
            float dx = xi - particles[j].x;
            float dy = yi - particles[j].y;

            float dvxji = particles[j].vx - vxi;
            float dvyji = particles[j].vy - vyi;

            float volume = particles[j].mass / particles[j].rho;

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
        particles[i].pxx = -particles[i].pressure;
        particles[i].pxy = 0.0f;
        particles[i].pyy = -particles[i].pressure;
} else if constexpr (IDEAL_FLUID == 2) {
        particles[i].pxx = -particles[i].pressure - _2_3_VISC * divv + _2VISC * dvx_dx;
        particles[i].pxy = VISC * (dvx_dy + dvy_dx);
        particles[i].pyy = -particles[i].pressure - _2_3_VISC * divv + _2VISC * dvy_dy;
}

    }
}

// force
void computeAcceleration(    
    std::vector<Particle>& particles,     
    const SpatialHashGrid& grid
) {
    int n = particles.size();
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float xi = particles[i].x;
        float yi = particles[i].y;

        float vxi = particles[i].vx;
        float vyi = particles[i].vy;
        
        float rhoi2 = particles[i].rho * particles[i].rho;

        float pxxi = particles[i].pxx;
        float pxyi = particles[i].pxy;
        float pyyi = particles[i].pyy;

        // acceleration
        float axi = 0.0f;
        float ayi = GRAV;

        grid.forEachNeighbour(i, particles, H, H2, [&](int j) {
            float dx = xi - particles[j].x;
            float dy = yi - particles[j].y;

            float mj = particles[j].mass;
            float rhoj2 = particles[j].rho * particles[j].rho;
            
            float pxxj = particles[j].pxx;
            float pxyj = particles[j].pxy;
            float pyyj = particles[j].pyy;

            auto [dW_dx, dW_dy] = get_dW_dxi(dx, dy);

            float term1 = (pxxi / rhoi2 + pxxj / rhoj2);
            float term2 = (pxyi / rhoi2 + pxyj / rhoj2);
            float term3 = (pyyi / rhoi2 + pyyj / rhoj2);
            
            axi += mj * (term1 * dW_dx + term2 * dW_dy);
            ayi += mj * (term2 * dW_dx + term3 * dW_dy);

            // monaghan artificial viscosity, avoiding from particles cross each other
            float dvx = particles[i].vx - particles[j].vx;
            float dvy = particles[i].vy - particles[j].vy;
            float vdotr = dvx * dx + dvy * dy;   // v_ij · r_ij

            if (vdotr < 0.0f) {
                float r = std::sqrt(dx*dx + dy*dy + 0.0001f * H2);
                float mu = H * vdotr / (r * r + 0.0001f * H2);
                float rhoAvg = 0.5f * (particles[i].rho + particles[j].rho);
                // alpha ~ 0.1, beta ~ 0.2, cs = speed of sound
                float pi_ij = (-MONAGHAN_ALPHA * CS * mu + MONAGHAN_BETA * mu * mu) / rhoAvg;
                
                axi -= mj * pi_ij * dW_dx;
                ayi -= mj * pi_ij * dW_dy;
            }
        });

        particles[i].ax = axi;
        particles[i].ay = ayi;
    }
}

const float DAMPING = -0.5f; 
void integrate(std::vector<Particle>& particles) {
    int n = particles.size();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        auto& p = particles[i];

        p.vx += p.ax * DT;
        p.vy += p.ay * DT;

        p.x += p.vx * DT;
        p.y += p.vy * DT;

        if (p.x < 0.0f) { 
            p.x = 0.0f;        
            p.vx *= DAMPING; 
        }
        else if (p.x > physicalWidth) {
            p.x = physicalWidth;
            p.vx *= DAMPING; 
        }
        if (p.y < 0.0f) {
            p.y = 0.0f;
            p.vy *= DAMPING; 
        }
        else if (p.y > physicalHeight) { 
            p.y = physicalHeight;
            p.vy *= DAMPING; 
        }
    }
}


int main() {

    std::cout << "simulation step time: DT " << DT << "\n";

    InitWindow(screenWidth, screenHeight, "SPH 2D Fluid Insight");

    // init particles
    std::vector<Particle> particles = getParticles();

    std::cout << "Total particles: " << particles.size() << std::endl;

    // arrange those paritcles in spatial hash struct
    SpatialHashGrid grid(physicalWidth, physicalHeight, H);

    Camera2D camera = { 0 };
    camera.zoom = 1.0f;

    SetTargetFPS(60);

    float simulatedTime = 0.0f;

    auto startTime = std::chrono::high_resolution_clock::now();
    int iteration = 0;

    while (!WindowShouldClose()) {
        // `substep` times iteration in one frame
        for(int i=0; i < substep; i++) {
            grid.build(particles);
            
            computeDensityPressure(particles, grid);
            computeStress(particles, grid);
            computeAcceleration(particles, grid);

            integrate(particles);

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

            for (const auto& p : particles) {
                float colorFactor = std::clamp(p.pressure / 500.0f, 0.0f, 1.0f);
                Color c = {(unsigned char)(255 * colorFactor), (unsigned char)(100 + 155 * (1-colorFactor)), 255, 200};
                // Color c = {(unsigned char)255, (unsigned char)0, (unsigned char)0, (unsigned char)255};
                // map the physical position to screen position
                DrawCircleV(Vector2{PHYSICAL_SCREEN_MAPPING * p.x, PHYSICAL_SCREEN_MAPPING * p.y}, PARTICLE_RADIUS, c); 
            }

        EndMode2D();
        DrawFPS(10, 10);
        DrawText(TextFormat("Particles: %d", particles.size()), 10, 35, 20, RAYWHITE);      
        
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