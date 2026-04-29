#pragma once

#include <vector>

#include "sph2d.h"

namespace sph2d {

namespace shape { // sph2d::shape

std::vector<Particle> generateParticles(
    std::function<float(float, float)> sdf,   
    float x0, float y0,                        
    float x1, float y1,                        
    float spacing = DX
) {
    std::vector<Particle> particles;

    for (float y = y0 + spacing * 0.5f; y < y1; y += spacing) {
        for (float x = x0 + spacing * 0.5f; x < x1; x += spacing) {
            if (sdf(x, y) < 0.0f) {
                Particle p;
                p.x = x;   p.y = y;
                p.vx = p.vy = p.ax = p.ay = 0.0f;
                p.rho = RHO0;
                p.mass = RHO0 * DX * DX;
                p.pressure = 0.0f;
                p.pxx = p.pxy = p.pyy = 0.0f;
                particles.emplace_back(p);
            }
        }
    }

    return particles;
}

std::vector<Particle> generateRect(
    float cx, float cy,    // center
    float w,  float h      // width and height
) {
    auto sdf = [=](float x, float y) {
        float dx = std::abs(x - cx) - w * 0.5f;
        float dy = std::abs(y - cy) - h * 0.5f;
        return std::max(dx, dy);
    };
    return generateParticles(sdf, cx - w*0.5f, cy - h*0.5f, cx + w*0.5f, cy + h*0.5f);
}

std::vector<Particle> generateCircle(
    float cx, float cy,    // center
    float radius
) {
    auto sdf = [=](float x, float y) {
        float dx = x - cx, dy = y - cy;
        return std::sqrt(dx*dx + dy*dy) - radius;
    };
    return generateParticles(sdf, cx - radius, cy - radius, cx + radius, cy + radius);
}

std::vector<Particle> generateTriangle(
    float x0, float y0,   // vertex A
    float x1, float y1,   // vertex B
    float x2, float y2    // vertex C
) {
    auto cross2d = [](float ax, float ay, float bx, float by) {
        return ax * by - ay * bx;
    };

    auto sdf = [=](float px, float py) {
        float d0 = cross2d(x1-x0, y1-y0, px-x0, py-y0);
        float d1 = cross2d(x2-x1, y2-y1, px-x1, py-y1);
        float d2 = cross2d(x0-x2, y0-y2, px-x2, py-y2);
        bool inside = (d0 >= 0 && d1 >= 0 && d2 >= 0)
                   || (d0 <= 0 && d1 <= 0 && d2 <= 0);
        return inside ? -1.0f : 1.0f;
    };

    float bx0 = std::min({x0, x1, x2});
    float by0 = std::min({y0, y1, y2});
    float bx1 = std::max({x0, x1, x2});
    float by1 = std::max({y0, y1, y2});
    return generateParticles(sdf, bx0, by0, bx1, by1);
}

} // namespace shp2d::shape

} // namespace sph2d