#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <stdint.h>
#include <atomic>

#include "sph2d.h"


namespace sph2d {

namespace collection {

struct SpatialHashGrid {
    float inv_cell;
    int   grid_w, grid_h, grid_n;

    std::vector<int> cell_count;   // number of particles in a grid
    std::vector<int> cell_start;   // offset
    std::vector<int> sorted_ids;   // sorted particle ids

    SpatialHashGrid(float domain_w, float domain_h, float h) {
        inv_cell = 1.0f / h;
        grid_w   = (int)std::ceil(domain_w * inv_cell) + 2;
        grid_h   = (int)std::ceil(domain_h * inv_cell) + 2;
        grid_n   = grid_w * grid_h;
        cell_count.resize(grid_n);
        cell_start.resize(grid_n);
    }

    inline int cellOf(float x, float y) const {
        int cx = std::clamp((int)(x * inv_cell), 0, grid_w - 1);
        int cy = std::clamp((int)(y * inv_cell), 0, grid_h - 1);
        return cy * grid_w + cx;
    }

    void build(const std::vector<Particle>& particles) {
        int n = (int)particles.size();
        sorted_ids.resize(n);

        // parallel count
        int n_threads;
        #pragma omp parallel
        { n_threads = omp_get_num_threads(); }

        std::vector<std::vector<int>> local_counts(n_threads, std::vector<int>(grid_n, 0));

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            int tid = omp_get_thread_num();
            local_counts[tid][cellOf(particles[i].x, particles[i].y)]++;
        }

        // merge local counts to global counts
        std::fill(cell_count.begin(), cell_count.end(), 0);
        for (int t = 0; t < n_threads; t++)
            for (int c = 0; c < grid_n; c++)
                cell_count[c] += local_counts[t][c];

        // prefix sum
        cell_start[0] = 0;
        for (int c = 1; c < grid_n; c++)
            cell_start[c] = cell_start[c-1] + cell_count[c-1];

        std::vector<std::atomic<int>> write_pos(grid_n);
        for (int c = 0; c < grid_n; c++)
            write_pos[c].store(cell_start[c], std::memory_order_relaxed);

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            int c   = cellOf(particles[i].x, particles[i].y);
            int pos = write_pos[c].fetch_add(1, std::memory_order_relaxed);
            sorted_ids[pos] = i;
        }
    }

    // read-only, parallelizable
    template<typename Func>
    inline void forEachNeighbour(
        int i,
        const std::vector<Particle>& particles,
        float h, float h2,
        Func callback
    ) const {
        float xi = particles[i].x;
        float yi = particles[i].y;

        int cx0 = std::max((int)((xi - h) * inv_cell),     0);
        int cy0 = std::max((int)((yi - h) * inv_cell),     0);
        int cx1 = std::min((int)((xi + h) * inv_cell) + 1, grid_w - 1);
        int cy1 = std::min((int)((yi + h) * inv_cell) + 1, grid_h - 1);

        for (int gy = cy0; gy <= cy1; gy++) {
            for (int gx = cx0; gx <= cx1; gx++) {
                int c     = gy * grid_w + gx;
                int start = cell_start[c];
                int end   = start + cell_count[c];
                for (int k = start; k < end; k++) {
                    int j = sorted_ids[k];
                    float dx = xi - particles[j].x;
                    float dy = yi - particles[j].y;
                    if (dx*dx + dy*dy < h2)
                        callback(j);
                }
            }
        }
    }
};

// template <typename T>
// struct alignas(64) PaddedAtomic {
//     std::atomic<T> val{0};

//     PaddedAtomic() = default;

//     PaddedAtomic(PaddedAtomic&& other) noexcept : val(other.val.load(std::memory_order_relaxed)) {}

//     PaddedAtomic& operator=(PaddedAtomic&& other) noexcept {
//         val.store(other.val.load(std::memory_order_relaxed), std::memory_order_relaxed);
//         return *this;
//     }

//     PaddedAtomic(const PaddedAtomic&) = delete;
//     PaddedAtomic& operator=(const PaddedAtomic&) = delete;
// };

// // for SoA
// // inplemented by atomic write position
// struct SpatialHashGridSoA {
//     float inv_cell;
//     int   grid_w, grid_h, grid_n;
//     int n_threads;                  // work on parallel threads

//     std::vector<int> cell_count;    // number of particles in a grid
//     std::vector<int> cell_start;    // offset
//     std::vector<int> sorted_ids;    // sorted particle ids
    
//     std::vector<int> local_counts_flat; // flattened local counts
//     std::vector<PaddedAtomic<int>> write_pos;

//     SpatialHashGridSoA(float domain_w, float domain_h, float h) {
//         inv_cell = 1.0f / h;
//         grid_w   = (int)std::ceil(domain_w * inv_cell) + 2;
//         grid_h   = (int)std::ceil(domain_h * inv_cell) + 2;
//         grid_n   = grid_w * grid_h;

//         cell_count.resize(grid_n);
//         cell_start.resize(grid_n);
//         write_pos.resize(grid_n);

//         #pragma omp parallel
//         #pragma omp single
//         n_threads = omp_get_num_threads();

//         local_counts_flat.resize(n_threads * grid_n, 0);
//     }

//     inline int cellOf(float x, float y) const {
//         int cx = std::clamp((int)(x * inv_cell), 0, grid_w - 1);
//         int cy = std::clamp((int)(y * inv_cell), 0, grid_h - 1);
//         return cy * grid_w + cx;
//     }

//     void build(const ParticleSystem& system) {
//         int n = (int)system.x.size();
//         sorted_ids.resize(n);

//         std::fill(local_counts_flat.begin(), local_counts_flat.end(), 0);

//         // parallel count
//         #pragma omp parallel for schedule(static)
//         for (int i = 0; i < n; i++) {
//             int tid = omp_get_thread_num();
//             local_counts_flat[tid * grid_n + cellOf(system.x[i], system.y[i])]++;
//         }

//         // merge local counts to global counts
//         std::fill(cell_count.begin(), cell_count.end(), 0);
//         for (int t = 0; t < n_threads; t++)
//             for (int c = 0; c < grid_n; c++)
//                 cell_count[c] += local_counts_flat[t * grid_n + c];

//         // prefix sum
//         cell_start[0] = 0;
//         for (int c = 1; c < grid_n; c++)
//             cell_start[c] = cell_start[c-1] + cell_count[c-1];

//         for (int c = 0; c < grid_n; c++)
//             write_pos[c].val.store(cell_start[c], std::memory_order_relaxed);

//         #pragma omp parallel for schedule(static)
//         for (int i = 0; i < n; i++) {
//             int c   = cellOf(system.x[i], system.y[i]);
//             int pos = write_pos[c].val.fetch_add(1, std::memory_order_relaxed);
//             sorted_ids[pos] = i;
//         }
//     }

//     // read-only, parallelizable
//     template<typename Func>
//     inline void forEachNeighbour(
//         int i,
//         const ParticleSystem& system,
//         float h, float h2,
//         Func callback
//     ) const {
//         float xi = system.x[i];
//         float yi = system.y[i];

//         int cx0 = std::max((int)((xi - h) * inv_cell),     0);
//         int cy0 = std::max((int)((yi - h) * inv_cell),     0);
//         int cx1 = std::min((int)((xi + h) * inv_cell) + 1, grid_w - 1);
//         int cy1 = std::min((int)((yi + h) * inv_cell) + 1, grid_h - 1);

//         for (int gy = cy0; gy <= cy1; gy++) {
//             for (int gx = cx0; gx <= cx1; gx++) {
//                 int c     = gy * grid_w + gx;
//                 int start = cell_start[c];
//                 int end   = start + cell_count[c];
//                 for (int k = start; k < end; k++) {
//                     int j = sorted_ids[k];
//                     float dx = xi - system.x[j];
//                     float dy = yi - system.y[j];
//                     if (dx*dx + dy*dy < h2)
//                         callback(j);
//                 }
//             }
//         }
//     }
// };


struct SpatialHashGridSoA {
    float inv_cell;
    int grid_w, grid_h, grid_n;
    int n_threads;

    std::vector<int> cell_count;        // number of particles in a grid
    std::vector<int> cell_start;        // offset
    std::vector<int> sorted_ids;        // sorted particle ids
    
    std::vector<int> local_counts_flat; // flattened local counts
    std::vector<int> thread_cell_start; // offset of each thread's cell start

    // temp buffer
    std::vector<float> tmp_x, tmp_y, tmp_vx, tmp_vy, tmp_ax, tmp_ay, 
                       tmp_mass, tmp_rho, tmp_pressure, tmp_pxx, tmp_pxy, tmp_pyy;

    SpatialHashGridSoA(float domain_w, float domain_h, float h) {
        inv_cell = 1.0f / h;
        grid_w   = (int)std::ceil(domain_w * inv_cell) + 2;
        grid_h   = (int)std::ceil(domain_h * inv_cell) + 2;
        grid_n   = grid_w * grid_h;

        n_threads = omp_get_max_threads(); 

        // or        
        // #pragma omp parallel
        // #pragma omp single
        // n_threads = omp_get_num_threads();

        cell_count.resize(grid_n);
        cell_start.resize(grid_n);
        local_counts_flat.resize(n_threads * grid_n);
        thread_cell_start.resize(n_threads * grid_n);
    }

    inline int cellOf(float x, float y) const {
        int cx = std::clamp((int)(x * inv_cell), 0, grid_w - 1);
        int cy = std::clamp((int)(y * inv_cell), 0, grid_h - 1);
        return cy * grid_w + cx;
    }

    void build(ParticleSystem& system) {
        int n = (int)system.x.size();
        sorted_ids.resize(n);

        // 1
        std::fill(local_counts_flat.begin(), local_counts_flat.end(), 0);

        // 2. parallel count particles
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int *local_counts = &local_counts_flat[tid * grid_n];

            #pragma omp for
            for (int i = 0; i < n; i++) {
                local_counts[cellOf(system.x[i], system.y[i])]++;
            }
        }

        // 3. parallel prefix sum
        int current_offset = 0;
        for (int c = 0; c < grid_n; c++) {
            cell_start[c] = current_offset;
            int total_in_cell = 0;
            for (int t = 0; t < n_threads; t++) {
                thread_cell_start[t * grid_n + c] = current_offset + total_in_cell;
                total_in_cell += local_counts_flat[t * grid_n + c];
            }
            cell_count[c] = total_in_cell;
            current_offset += total_in_cell;
        }

        // 4. parallel fill sorted_ids
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            int tid = omp_get_thread_num();
            int c   = cellOf(system.x[i], system.y[i]);
            int dest_idx = thread_cell_start[tid * grid_n + c]++;
            sorted_ids[dest_idx] = i;
        }

        // 5. physical reordering
        reorderParticleSystem(system);
    }

    void resizeBuffers(int n) {
        tmp_x.resize(n); tmp_y.resize(n); tmp_vx.resize(n); tmp_vy.resize(n);
        tmp_ax.resize(n); tmp_ay.resize(n); tmp_mass.resize(n);
        tmp_rho.resize(n); tmp_pressure.resize(n);
        tmp_pxx.resize(n); tmp_pxy.resize(n); tmp_pyy.resize(n);
    }

    // for better cache hit
    void reorderParticleSystem(ParticleSystem& system) {
        int n = (int)system.x.size();
        resizeBuffers(n);

        #pragma omp parallel for schedule(static)
        for (int new_idx = 0; new_idx < n; new_idx++) {
            int old_idx = sorted_ids[new_idx];

            tmp_x[new_idx]        = system.x[old_idx];
            tmp_y[new_idx]        = system.y[old_idx];
            tmp_vx[new_idx]       = system.vx[old_idx];
            tmp_vy[new_idx]       = system.vy[old_idx];
            tmp_ax[new_idx]       = system.ax[old_idx];
            tmp_ay[new_idx]       = system.ay[old_idx];
            tmp_mass[new_idx]     = system.mass[old_idx];
            tmp_rho[new_idx]      = system.rho[old_idx];
            tmp_pressure[new_idx] = system.pressure[old_idx];
            tmp_pxx[new_idx]      = system.pxx[old_idx];
            tmp_pxy[new_idx]      = system.pxy[old_idx];
            tmp_pyy[new_idx]      = system.pyy[old_idx];
        }

        system.x.swap(tmp_x);
        system.y.swap(tmp_y);
        system.vx.swap(tmp_vx);
        system.vy.swap(tmp_vy);
        system.ax.swap(tmp_ax);
        system.ay.swap(tmp_ay);
        system.mass.swap(tmp_mass);
        system.rho.swap(tmp_rho);
        system.pressure.swap(tmp_pressure);
        system.pxx.swap(tmp_pxx);
        system.pxy.swap(tmp_pxy);
        system.pyy.swap(tmp_pyy);
    }

    template<typename Func>
    inline void forEachNeighbour(
        int i,
        const ParticleSystem& system,
        float h, float h2,
        Func callback
    ) const {
        float xi = system.x[i];
        float yi = system.y[i];

        int cx0 = std::max((int)((xi - h) * inv_cell),     0);
        int cy0 = std::max((int)((yi - h) * inv_cell),     0);
        int cx1 = std::min((int)((xi + h) * inv_cell) + 1, grid_w - 1);
        int cy1 = std::min((int)((yi + h) * inv_cell) + 1, grid_h - 1);

        for (int gy = cy0; gy <= cy1; gy++) {
            for (int gx = cx0; gx <= cx1; gx++) {
                int c     = gy * grid_w + gx;
                int start = cell_start[c];
                int end   = start + cell_count[c];
                for (int k = start; k < end; k++) {
                    float dx = xi - system.x[k];
                    float dy = yi - system.y[k];
                    if (dx*dx + dy*dy < h2)
                        callback(k);
                }
            }
        }
    }

    // // process 16 particles in a time by callback16, and remaining particles process by callback1
    // template <typename Func16, typename Func1>
    // inline void forEachNeighbour16(
    //     int i,
    //     const ParticleSystem& system,
    //     float h, float h2,
    //     Func16 callback16,
    //     Func1 callback1
    // ) const {
    //     float xi = system.x[i];
    //     float yi = system.y[i];
    // }

    std::array<int, 9> getNeighbourCells(int c) const {
        int c_minus_grid_w = c - grid_w;
        int c_plus_grid_w = c + grid_w;
        return {
            c_minus_grid_w - 1, c_minus_grid_w, c_minus_grid_w + 1,     // top line
            c - 1,              c,               c + 1,                 // curr line
            c_plus_grid_w - 1,  c_plus_grid_w,  c_plus_grid_w + 1       // bottom line
        };
    }

    // ! Extension for Verlet List
    static const int MAX_NEIGHBOURS = 64;
    std::vector<int> neighbour_count;

    std::vector<int> neighbour_list_flat;

    void initVerletList(int num_particles) {
        neighbour_count.resize(num_particles, 0);
        neighbour_list_flat.resize(num_particles * MAX_NEIGHBOURS, 0);
    }

    // build verlet list
    // after grid.build(system)
    void buildVerletList(const ParticleSystem& system, float search_radius) {
        int n = system.x.size();

        if (neighbour_count.size() != n) {
            initVerletList(n);
        }

        #pragma omp parallel for schedule(dynamic, DYNAMIC_SCHEDULE_PARTICLE_BASED)
        for (int i = 0; i < n; i++) {
            int count = 0;

            int base_idx = i * MAX_NEIGHBOURS;

            forEachNeighbour(i, system, search_radius, search_radius * search_radius, [&](int j) {
                if (count < MAX_NEIGHBOURS) {
                    neighbour_list_flat[base_idx + count++] = j;
                }
                // if exceed, should throw some Exception or allocate a larger memory pool
            });

            neighbour_count[i] = count;
        }
    }
};

} // namespace sph2d::collection

} // namespace sph2d