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
#include "renderer.h"

namespace feh {

class DFTracker {
public:
    DFTracker(const cv::Mat &img, const cv::Mat &edge,
            const Vec2i &shape, 
            ftype fx, ftype fy, ftype cx, ftype cy,
            const SE3 &g,
            const MatX &V, const MatXi &F);

    /// \brief: Update image and edge evidence from external source.
    /// \param img: RGB image.
    /// \param edge: Edge map extracted from the image.
    void UpdateImage(const cv::Mat &img, const cv::Mat &edge);
    /// \brief: Update camera pose from external source (VIO).
    /// \param gsc: Camera to spatial frame transformation.
    void UpdateCameraPose(const SE3 &gsc);

    /// \brief: Minimzing step.
    ftype Minimize(int steps);

    /// \brief: Compute the loss at the current pose with given perturbation
    /// returns: residual vector and Jacobian matrix.
    std::tuple<VecX, MatX> ComputeResidualAndJacobian(const SE3 &g);
    /// \brief: Render at current pose estimate.
    cv::Mat RenderEstimate() const;
    /// \brief: Render edge pixels at current estimate.
    cv::Mat RenderEdgepixels() const ;
    /// \brief: Get current estimate of object pose.
    SE3 GetEstimate() const { return g_; }
    /// \brief: Get current distance field.
    cv::Mat GetDistanceField() const { return DistanceTransform::BuildView(DF_) ; }
    /// \brief: Get distance field gradients.
    cv::Mat GetDFGradient() const { return dDF_dxy_; }

protected:
    /// \brief: Build distance field from probabilitic edge map.
    void BuildDistanceField();

protected:
    RendererPtr engine_;
    cv::Mat img_, edge_, DF_;   // RGB, edge map, distance field
    cv::Mat dDF_dxy_;
    Vec2i shape_;
    Mat3 K_, Kinv_;
    SE3 g_; // object -> spatial frame
    SE3 gsc_; // camera -> spatial frame
    MatX V_;
    MatXi F_;
    Timer timer_;
};

}

