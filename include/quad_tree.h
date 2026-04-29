#pragma once

namespace sph2d {

namespace collection {

struct QuadTree {
    static constexpr int MAX_DEPTH      = 8; // max depth
    static constexpr int MAX_LEAF_CAP   = 8; // the maximum number of particles a leaf contains

    struct Node {
        float cx, cy, half;

    };
    

};

} // namespace sph2d::collectionb

} // namespace sph2d
