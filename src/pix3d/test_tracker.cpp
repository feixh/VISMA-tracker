#include <iostream>

#include "glog/logging.h"
#include "folly/Format.h"

#include "eigen_alias.h"
#include "utils.h"
#include "rodrigues.h"
#include "distance_transform.h"
#include "renderer.h"
#include "pix3d/dataloader.h"

constexpr float znear = 0.1;
constexpr float zfar = 10;

namespace feh {
class DiffTracker {
public:
    DiffTracker(const cv::Mat &img, const cv::Mat &edge,
            const Vec2i &shape, 
            ftype fx, ftype fy, ftype cx, ftype cy,
            const Mat3 &R, const Vec3 &T,
            const MatX &V, const MatXi &F):
    _img(img.clone()), _edge(edge.clone()), 
    _shape(shape), 
    _R(R), _T(T), _V(V), _F(F) 
    {
        _K << fx, 0, cx,
            0, fy, cy,
            0, 0, 1;
        Mat3 flip = Mat3::Zero();
        flip << -1, 0, 0,
             0, -1, 0,
             0, 0, 1;
        _R = flip * _R;
        _T = flip * _T;
        _engine = std::make_shared<Renderer>(_shape[0], _shape[1]);
        _engine->SetCamera(znear, zfar, fx, fy, cx, cy);

        cv::Mat normalized_edge;
        normalized_edge = cv::Scalar::all(255) - _edge;
        cv::distanceTransform(normalized_edge / 255.0, _DF, CV_DIST_L2, CV_DIST_MASK_PRECISE);

        _dW.setZero();
        _dT.setZero();
    }

    cv::Mat Minimize() {
        ftype stepsize = eps;
        cv::namedWindow("iter", CV_WINDOW_NORMAL);
        for (int iter = 0; iter < 100; ++iter) {
            std::cout << "==========\n";
            std::cout << "iter=" << iter << std::endl;
            Vec3 dC_dW, dC_dT;
            ftype C = ComputeLoss(dC_dW, dC_dT);
            _R -= hat(Vec3{stepsize * dC_dW});
            _T -= stepsize * dC_dT;
            std::cout << "Cost=" << C << std::endl;

            cv::Mat depth = RenderCurrentEstimate();
            cv::imshow("iter", depth);
            cv::waitKey(10);

        }
        return RenderCurrentEstimate();
    }

    ftype ComputeLoss(Vec3 &dC_dW, Vec3 &dC_dT) {
        ftype C0 = ForwardPass(Vec3{Vec3::Zero()}, Vec3{Vec3::Zero()});
        // Vec3 dC_dW = Vec3::Zero();
        // Vec3 dC_dT = Vec3::Zero();
        dC_dW.setZero();
        dC_dT.setZero();
        for (int i = 0; i < 3; ++i) {
            Vec3 dW = Vec3::Zero();
            dW(i) = eps;
            dC_dW(i) = (ForwardPass(dW, Vec3{Vec3::Zero()}) - C0) / eps;

            Vec3 dT = Vec3::Zero();
            dT(i) = eps;
            dC_dT(i) = (ForwardPass(Vec3{Vec3::Zero()}, dT) - C0) / eps;
        }
        std::cout << "dC_dW=" << dC_dW.transpose() << std::endl;
        std::cout << "dC_dT=" << dC_dT.transpose() << std::endl;
        return C0;
    }

    /// \brief: Compute the loss at the current pose with given perturbation
    ftype ForwardPass(const Vec3 &dW, const Vec3 &dT) {
        Mat3 Rp = _R + _R * hat(dW);
        Vec3 Tp = _T + dT;
        auto V = TransformShape(Rp, Tp);
        // std::cout << V << std::endl;
        _engine->SetMesh((float*)V.data(), V.rows(), (int*)_F.data(), _F.rows());
        Eigen::Matrix<ftype, 4, 4, Eigen::ColMajor> identity;
        identity.setIdentity();
        std::vector<EdgePixel> edgelist;
        _engine->ComputeEdgePixels(identity, edgelist);
        ftype cost = 0;
        for (const auto &e : edgelist) {
            int row = std::max(0, std::min((int)e.y, _shape[0]-1));
            int col = std::max(0, std::min((int)e.x, _shape[1]-1));
            cost += _DF.at<float>(row, col);
        }
        cost /= edgelist.size();
        return cost;
    }

    /// \brief: Render at current pose estimate.
    cv::Mat RenderCurrentEstimate() {
        auto V = TransformShape(_R, _T);
        _engine->SetMesh((float*)V.data(), V.rows(), (int*)_F.data(), _F.rows());
        Eigen::Matrix<ftype, 4, 4, Eigen::ColMajor> identity;
        identity.setIdentity();
        cv::Mat depth(_shape[0], _shape[1], CV_32FC1); 
        _engine->RenderDepth(identity, depth);
        return depth;
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
    Vec2i _shape;
    Mat3 _K;
    Vec3 _dW, _dT;  // rotation and translation vector
    Mat3 _R;
    Vec3 _T;
    MatX _V;
    MatXi _F;
};

}


int main(int argc, char **argv) {
    CHECK_EQ(argc, 2) << "requires root directory of pix3d as an argument!";
    feh::Pix3dLoader loader(argv[1]);
    // auto packet = loader.GrabPacket("img/bed/0010.png"); // index by path
    // OR index by id
    auto packet = loader.GrabPacket(0); // index by path

    cv::namedWindow("image", CV_WINDOW_NORMAL);
    cv::imshow("image", packet._img);

    cv::namedWindow("mask", CV_WINDOW_NORMAL);
    cv::imshow("mask", packet._mask);

    cv::namedWindow("edge", CV_WINDOW_NORMAL);
    cv::imshow("edge", packet._edge);

    // noise generators
    auto generator = std::make_shared<std::knuth_b>();
    std::normal_distribution<float> normal_dist;
    float Tnoise = 0.5;
    float Rnoise = 0.1;

    feh::Vec3 Tn = packet._go.translation() 
        + Tnoise * feh::RandomVector<3>(0, Tnoise, generator);

    feh::Vec3 Wn = packet._go.so3().log() 
        + Rnoise * feh::RandomVector<3>(0, Rnoise, generator);

    feh::Mat3 Rn = rodrigues(Wn);


    feh::DiffTracker tracker(packet._img, packet._edge,
            packet._shape, 
            packet._focal_length, packet._focal_length,
            packet._shape[1] >> 1, packet._shape[0] >> 1,
            Rn, Tn,
            packet._V, packet._F);
    auto depth = tracker.Minimize();
    cv::namedWindow("depth", CV_WINDOW_NORMAL);
    cv::imshow("depth", depth);
    cv::waitKey();

}

