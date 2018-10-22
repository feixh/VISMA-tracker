#pragma once

#include <iostream>
#include <tuple>
#include <chrono>

#include "glog/logging.h"
#include "folly/Format.h"
#include "opencv2/imgproc.hpp"

#include "eigen_alias.h"
#include "utils.h"
#include "rodrigues.h"
#include "distance_transform.h"
#include "renderer.h"

namespace feh {

class DiffTracker {
public:
    DiffTracker(const cv::Mat &img, const cv::Mat &edge,
            const Vec2i &shape, 
            ftype fx, ftype fy, ftype cx, ftype cy,
            const Mat3 &R, const Vec3 &T,
            const MatX &V, const MatXi &F);

    ftype Minimize(int steps);

    /// \brief: Compute the loss at the current pose with given perturbation
    /// returns: residual vector and Jacobian matrix.
    std::tuple<VecX, MatX> ComputeResidualAndJacobian(const Mat3 &R, const Vec3 &T);

    /// \brief: Render at current pose estimate.
    cv::Mat RenderEstimate() const;
    /// \brief: Render edge pixels at current estimate.
    cv::Mat RenderEdgepixels() const ;
    std::tuple<Mat3, Vec3> GetEstimate() const { return std::make_tuple(_R, _T); }
    cv::Mat GetDistanceField() const { return _DF; }
    std::tuple<cv::Mat, cv::Mat> GetDFGradient() const { return std::make_tuple(_dDF_dx, _dDF_dy); }


    /// \brief: Transform the mesh according to the given pose.
    MatX TransformShape(const Mat3 &R, const Vec3 &T) const {
        MatX V(_V.rows(), _V.cols());
        V.setZero();
        for (int i = 0; i < _V.rows(); ++i) {
            V.row(i) = R * _V.row(i).transpose() + T;
        }
        return V;
    }


private:
    RendererPtr _engine;
    cv::Mat _img, _edge, _DF;   // RGB, edge map, distance field
    cv::Mat _dDF_dx, _dDF_dy;
    Vec2i _shape;
    Mat3 _K, _Kinv;
    Mat3 _R;
    Vec3 _T;
    MatX _V;
    MatXi _F;
    Timer _timer;
};

}

