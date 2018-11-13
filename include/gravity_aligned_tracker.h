#pragma once

#include <iostream>
#include <tuple>
#include <chrono>

#include "glog/logging.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "absl/strings/str_format.h"

#include "alias.h"
#include "rodrigues.h"
#include "utils.h"
#include "distance_transform.h"
#include "DFtracker.h"
#include "renderer.h"
#include "se3.h"

namespace feh {

class GravityAlignedTracker: public DFTracker {
public:
    GravityAlignedTracker(const cv::Mat &img, const cv::Mat &edge,
            const Vec2i &shape, 
            ftype fx, ftype fy, ftype cx, ftype cy,
            const SE3 &g,
            const MatX &V, const MatXi &F):
        DFTracker(img, edge, shape, fx, fy, cx, cy, g, V, F) {
            UpdateGravity(SO3{});
        }

    ftype Minimize(int steps);
    std::tuple<VecX, MatX> ComputeResidualAndJacobian(const SE3 &g);
    void UpdateGravity(const SO3 &Rg);

protected:
    SO3 Rg_;
    Vec3 gamma_;
};

}

