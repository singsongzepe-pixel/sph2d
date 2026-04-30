#pragma once

#include <cmath>
#include <array>
#include <raylib.h>

#ifdef _MSC_VER
    #define FORCE_INLINE __forceinline inline
#else
    #define FORCE_INLINE __attribute__((always_inline)) inline
#endif

// already defined in raylib
// #define PI 3.1415926535f

namespace sph2d {

const float RHO0        = 1000.0f;      // density of water
const float sideLen     = 1.0f;         // simulate 1m square
const int N_side        = 100;          // number of particle in the side
const float DX          = sideLen / (float)N_side; // the original distance between particles
const float H           = 3.0f * DX;
const float H2          = H*H;
const float CS          = 50.0f;

const float VISC        = 1.01e-3f;
const float GRAV        = 9.8f;        // in gravity field
const float DT          = H / CS / 5.0f;

const float MONAGHAN_ALPHA  = 0.2f;
const float MONAGHAN_BETA   = 0.0f;

// preformance constant
const float _2_3_VISC   = 2.0f / 3.0f * VISC;
const float _2VISC      = 2.0f * VISC;

// 4 x 12 = 48 bytes
// one cache line (64 bytes) cannot even hold two particles
struct Particle { 
    float x, y;
    float vx, vy;
    float ax, ay;
    float mass;
    float rho, pressure;
    float pxx, pxy, pyy; // stress
};

// SoA
struct ParticleSystem {
    std::vector<float> x, y;
    std::vector<float> vx, vy;
    std::vector<float> ax, ay;
    std::vector<float> mass;
    std::vector<float> rho, pressure;
    std::vector<float> pxx, pxy, pyy;
    
    ParticleSystem(const std::vector<Particle>& particles) {
        int n = particles.size();
        
        x.reserve(n);
        y.reserve(n);
        
        vx.reserve(n);
        vy.reserve(n);
        
        ax.reserve(n);
        ay.reserve(n);

        mass.reserve(n);
        rho.reserve(n);
        pressure.reserve(n);

        pxx.reserve(n);
        pxy.reserve(n);
        pyy.reserve(n);

        for (const auto& p : particles) {
            x.emplace_back(p.x);
            y.emplace_back(p.y);

            vx.emplace_back(p.vx);
            vy.emplace_back(p.vy);

            ax.emplace_back(p.ax);
            ay.emplace_back(p.ay);

            mass.emplace_back(p.mass);
            rho.emplace_back(p.rho);
            pressure.emplace_back(p.pressure);
            
            pxx.emplace_back(p.pxx);
            pxy.emplace_back(p.pxy);
            pyy.emplace_back(p.pyy);
        }
    }
};

const float alpha = 5.0f / (PI * H2);
// according to pro-neighbours finding, the r always less than H
// LucyQuartic kernel function
float get_W(float r) { // r - real distance between two particles
    float R = r / H;
    float R2 = R*R;
    float R3 = R2*R;
    float R4 = R3*R;

    return alpha * (1 - 6*R2 + 8*R3 - 3*R4);
}

const float beta = -12.0f * alpha / H2;
std::array<float, 2> get_dW_dxi(float dx, float dy) {    
    float r = std::sqrt(dx*dx + dy*dy);
    float term = 1.0f - r/H;

    float common = beta * term * term;
    return {
        common * dx,
        common * dy,
    };
}

// some other kernel function
// POLY6 kernel function
const float alpha_poly6 = 4.0f / (PI * std::pow(H, 8));
FORCE_INLINE float get_W_poly6(float r2) {
    float term = H2 - r2;
    return alpha_poly6 * (term * term * term);
}

const float beta_poly6 = -24.0f / (PI * std::pow(H, 8));
FORCE_INLINE std::array<float, 2> get_dW_dxi_poly6(float dx, float dy) {
    float r2 = dx*dx + dy*dy;
    float term = H2 - r2;

    float common = beta_poly6 * (term * term);
    return {
        common * dx,
        common * dy,
    };
}

// simd version of POLY6
FORCE_INLINE __m512 get_W_poly6_simd(__m512 v_r2, __m512 v_H2, __m512 v_factor) {
    __m512 diff = _mm512_sub_ps(v_H2, v_r2);
    __m512 diff2 = _mm512_mul_ps(diff, diff);
    __m512 diff3 = _mm512_mul_ps(diff2, diff);

    return _mm512_mul_ps(v_factor, diff3);
}

FORCE_INLINE void get_dW_dxi_poly6_simd(
    __m512 v_dx, __m512 v_dy, 
    __m512 v_r2, __m512 v_H2, 
    __m512 v_f_grad,
    __m512& v_dW_dx, __m512& v_dW_dy,
    __mmask16 v_mask
) {
    __m512 v_diff = _mm512_sub_ps(v_H2, v_r2);
    __m512 v_diff2 = _mm512_mul_ps(v_diff, v_diff);
    __m512 v_common = _mm512_mul_ps(v_f_grad, v_diff2);
    
    v_dW_dx = _mm512_mul_ps(v_common, v_dx);
    v_dW_dy = _mm512_mul_ps(v_common, v_dy);    
}

const float GAMMA = 7;
const float B = 30000.0f;               // stiffness
float get_pressure(float rho) {
    float p = B * (std::pow(rho / RHO0, GAMMA) - 1.0f);
    return p;
}

} // namespace sph2d
