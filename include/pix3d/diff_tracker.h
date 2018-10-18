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

    std::tuple<VecX, MatX> ComputeLoss();
    /// \brief: Compute the loss at the current pose with given perturbation
    std::tuple<VecX, VecX> ForwardPass(const Vec3 &dW, const Vec3 &dT, 
            Eigen::Matrix<ftype, Eigen::Dynamic, 3> &X);

    std::tuple<VecX, MatX> ComputeLoss2();

    /// \brief: Render at current pose estimate.
    cv::Mat RenderEstimate() {
        auto V = TransformShape(_R, _T);
        _engine->SetMesh((float*)V.data(), V.rows(), (int*)_F.data(), _F.rows());
        Eigen::Matrix<ftype, 4, 4, Eigen::ColMajor> identity;
        identity.setIdentity();
        cv::Mat depth(_shape[0], _shape[1], CV_32FC1); 
        _engine->RenderDepth(identity, depth);
        return depth;
    }

    std::tuple<Mat3, Vec3> GetEstimate() {
        return std::make_tuple(_R, _T);
    }

    cv::Mat GetDistanceField() {
        return _DF;
    }

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
    cv::Mat _dDFx, _dDFy;
    Vec2i _shape;
    Mat3 _K, _Kinv;
    Mat3 _R;
    Vec3 _T;
    MatX _V;
    MatXi _F;
};

}

