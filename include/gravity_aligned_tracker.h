#pragma once

#include <iostream>
#include <tuple>
#include <chrono>

#include "glog/logging.h"
#include "folly/Format.h"
#include "opencv2/imgproc.hpp"
#include "sophus/se3.hpp"

#include "eigen_alias.h"
#include "rodrigues.h"
#include "utils.h"
#include "distance_transform.h"
#include "DFtracker.h"
#include "renderer.h"

namespace feh {

class GravityAlignedTracker: public DFTracker {
public:
    GravityAlignedTracker(const cv::Mat &img, const cv::Mat &edge,
            const Vec2i &shape, 
            ftype fx, ftype fy, ftype cx, ftype cy,
            const SE3 &g,
            const MatX &V, const MatXi &F):
        DFTracker(img, edge, shape, fx, fy, cx, cy, g, V, F) {}
};

}

