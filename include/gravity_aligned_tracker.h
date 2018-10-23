#pragma once

#include <iostream>
#include <tuple>
#include <chrono>

#include "glog/logging.h"
#include "folly/Format.h"
#include "opencv2/imgproc.hpp"

#include "eigen_alias.h"
#include "rodrigues.h"
#include "utils.h"
#include "distance_transform.h"
#include "renderer.h"

namespace feh {

class GravityAlignedTracker {
public:
    GravityAlignedTracker(const cv::Mat &img, const cv::Mat &edge,
            const Vec2i &shape, 
            ftype fx, ftype fy, ftype cx, ftype cy,
            const Mat3 &R, const Vec3 &T,
            const MatX &V, const MatXi &F);

    /// \brief: Upload image and edge evidence.
    void Upload(const cv::Mat &img, const cv::Mat &edge) {
        img_ = img.clone();
        edge_ = edge.clone();
        BuildDistanceField();
    }


    /// \brief: Apply a rigid pose transformation to the state
    /// when the FOV changes.
    void ChangeReference(const Eigen::Matrix<ftype, 3, 4> &g_new_old) {
        R_ = g_new_old.leftCols(3) * R_;
        T_ = g_new_old.leftCols(3) * T_ + g_new_old.rightCols(1);
    }
    /// \brief: Minimzing step.
    ftype Minimize(int steps);

    /// \brief: Compute the loss at the current pose with given perturbation
    /// returns: residual vector and Jacobian matrix.
    std::tuple<VecX, MatX> ComputeResidualAndJacobian(const Mat3 &R, const Vec3 &T);
    /// \brief: Render at current pose estimate.
    cv::Mat RenderEstimate() const;
    /// \brief: Render edge pixels at current estimate.
    cv::Mat RenderEdgepixels() const ;
    /// \brief: Get current estimate of object pose.
    std::tuple<Mat3, Vec3> GetEstimate() const {
        return std::make_tuple(R_, T_); }
    /// \brief: Get current distance field.
    cv::Mat GetDistanceField() const { return DistanceTransform::BuildView(DF_) ; }
    /// \brief: Get distance field gradients.
    cv::Mat GetDFGradient() const { return dDF_dxy_; }


private:
    /// \brief: Build distance field from probabilitic edge map.
    void BuildDistanceField();
    /// \brief: Transform the mesh according to the given pose.
    MatX TransformShape(const Mat3 &R, const Vec3 &T) const {
        MatX V = V_ * R.transpose();
        V.rowwise() += T.transpose();
        return V;
    }


private:
    RendererPtr engine_;
    cv::Mat img_, edge_, DF_;   // RGB, edge map, distance field
    cv::Mat dDF_dxy_;
    Vec2i shape_;
    Mat3 K_, Kinv_;
    // object -> spatial frame
    Mat3 R_;
    Vec3 T_;
    // camera -> spatial frame
    Mat3 Rsc_;
    Vec3 Tsc_;
    MatX V_;
    MatXi F_;
    Timer timer_;
};

}

