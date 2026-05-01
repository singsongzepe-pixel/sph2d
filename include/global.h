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

#define DYNAMIC_SCHEDULE_CELL_BASED 64
#define DYNAMIC_SCHEDULE_PARTICLE_BASED 256

// ! SOME SIMULATION OPTIONS
// rho calculation formula
// 1. basic
// 2. Shepard normalized
#define RHO_VAR 1

// fluid feature
// 1: ideal fliud
// 2: real fluid
#define IDEAL_FLUID 2

// select test type
// 1. rectangle
// 2. circle
// 3. triangle
// 4. two circles collision
#define PARTICLE_SHAPE 4

// use reciprocal instruction to replace division
// 1. disable reciprocal instruction
// 2. enable reciprocal instruction
#define RECIPROCAL_REPLACEMENT 2

// boundary process
// 1. basic
// 2. penalty force
#define BOUNDARY_PROCESS 1

// software prefetch
// 1. disable software prefetch
// 2. enable software prefetch
#define SOFTWARE_PREFETCH 2
#define SOFTWARE_PREFETCH_DIST 64 // 32/64 don't be too late or too early


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

// basic boundary damping for benchmark, v1, v2, v3
const float DAMPING = -0.5f; 
const float BOUNDARY_SHIFT_EPSILON  = 0.003f;

// better boundary process for v4
const float PARTICLE_COLLISION_RADIUS = DX * 0.8f; 
const float BOUNDARY_STIFFNESS = 100000.0f; 
const float BOUNDARY_DAMPING   = 250.0f;
const float HARD_DAMPING       = -0.5f;

