#pragma once

#include "sph2d.h"
#include "shape.h"

// define the universal parameter for each version

// assuming the simulation area(bounding box) is a square
const float physicalWidth       = 3.0f; // [0.0f, ...f]
const float physicalHeight      = 3.0f; // [0.0f, ...f]
const int screenWidth           = 1200;
const int screenHeight          = 1200;
const float PHYSICAL_SCREEN_MAPPING = (float)screenWidth / physicalWidth;

// rho calculation formula
#define RHO_VAR 1

// 1: ideal fliud, 2: real fluid
#define IDEAL_FLUID 1

// select test type
#define PARTICLE_SHAPE 4
// pressure

using namespace sph2d;
using namespace sph2d::shape;

std::vector<Particle> getParticles() {
    std::vector<Particle> particles;

    if constexpr (PARTICLE_SHAPE == 1) {
        particles = generateRect(1.0f, 1.0f, 0.4f, 0.4f);
    } else if constexpr (PARTICLE_SHAPE == 2) {
        particles = generateCircle(1.0f, 1.0f, 0.2f);
    } else if constexpr (PARTICLE_SHAPE == 3) {
        particles = generateTriangle(
            0.8f, 0.8f,
            1.2f, 1.5f,
            1.3f, 0.8f
        );
    } else if constexpr (PARTICLE_SHAPE == 4) {
        auto p1 = generateCircle(0.6f, 1.0f, 0.3f);
        auto p2 = generateCircle(1.3f, 1.0f, 0.3f);

        for (auto& p : p2) p.vx = -2.0f;

        p1.insert(p1.end(), p2.begin(), p2.end());
        particles = p1;
    }

    return particles;
}

// substep in each frame
const int substep = 10;

// iteration to count time (test performance)
const int ITERATION_TO_COUNT = 100;

// visual
const float PARTICLE_RADIUS = 1.5f;

