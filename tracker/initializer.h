//
// Created by visionlab on 1/18/18.
//
#pragma once

#include "alias.h"
#include "utils.h"
#include "message_utils.h"

namespace feh {
/// \brief: solve absolute pose given 3D control points (X) and their 2D projections (x)
SE3 Initialize(const std::vector<Vec3> &X, const std::vector<Vec2> &x, const Mat3 &K=Mat3::Identity());

}   // namespace feh
