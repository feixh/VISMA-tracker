#include <iostream>
#include <tuple>

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
        _Kinv = _K.inverse();
        Mat3 flip = Mat3::Zero();
        flip << -1, 0, 0,
             0, -1, 0,
             0, 0, 1;
        _R = flip * _R;
        _T = flip * _T;
        _engine = std::make_shared<Renderer>(_shape[0], _shape[1]);
        _engine->SetCamera(znear, zfar, fx, fy, cx, cy);

        cv::Mat normalized_edge, tmp;
        normalized_edge = cv::Scalar::all(255) - _edge;
        cv::distanceTransform(normalized_edge / 255.0, tmp, CV_DIST_L2, CV_DIST_MASK_PRECISE);
        _DF = DistanceTransform::BuildView(tmp) / 255.0 * 10;    // DF value range [0, 1]
    }

    ftype Minimize(int steps=1) {
        ftype stepsize = 2.0;
        VecX r;
        MatX J;
        for (int iter = 0; iter < steps; ++iter) {
            std::tie(r, J) = ComputeLoss();

            // Gauss-Newton update
            MatX JtJ = J.transpose() * J;
            MatX damping(JtJ.rows(), JtJ.cols());
            damping.setIdentity();
            damping *= 0;
            Eigen::Matrix<ftype, 6, 1> delta = (JtJ + damping).ldlt().solve(J.transpose() * r);
            // _R = _R + hat<ftype>(delta.head<3>());
            _R = _R * rodrigues(Vec3{delta.head<3>()});
            _T = _T + delta.tail<3>();
        }
        return r.sum();
    }

    std::tuple<VecX, MatX> ComputeLoss() {
        Eigen::Matrix<ftype, Eigen::Dynamic, 3> X;

        VecX r, v;
        VecX rp, vp;

        std::tie(r, v) = ForwardPass(Vec3{Vec3::Zero()}, Vec3{Vec3::Zero()}, X);

        MatX J(X.rows(), 6);
        J.setZero();

        for (int i = 0; i < 3; ++i) {
            Vec3 dW = Vec3::Zero();
            dW(i) = eps;
            std::tie(rp, vp) = ForwardPass(dW, Vec3{Vec3::Zero()}, X);
            J.col(i) = vp.cwiseProduct(v.cwiseProduct(rp - r)) / eps;

            Vec3 dT = Vec3::Zero();
            dT(i) = eps;
            std::tie(rp, vp) = ForwardPass(Vec3{Vec3::Zero()}, dT, X);
            J.col(3+i) = vp.cwiseProduct(v.cwiseProduct(rp - r)) / eps;
        }
        return std::make_tuple(r, J);
    }

    /// \brief: Compute the loss at the current pose with given perturbation
    std::tuple<VecX, VecX> ForwardPass(const Vec3 &dW, const Vec3 &dT, 
            Eigen::Matrix<ftype, Eigen::Dynamic, 3> &X) {

        // perturbated pose
        Mat3 Rp = _R * rodrigues(dW);
        Vec3 Tp = _T + dT;

        VecX r, v;  // residual and valid bit

        if (X.size() == 0) {
            // no 3D points yet
            auto V = TransformShape(Rp, Tp);
            _engine->SetMesh((float*)V.data(), V.rows(), (int*)_F.data(), _F.rows());
            Eigen::Matrix<ftype, 4, 4, Eigen::ColMajor> identity;
            identity.setIdentity();
            std::vector<EdgePixel> edgelist;
            _engine->ComputeEdgePixels(identity, edgelist);

            r.resize(edgelist.size());
            X.resize(edgelist.size(), 3);
            v.resize(edgelist.size());
            v.setOnes();

            ftype cost = 0;
            for (int i = 0; i < edgelist.size(); ++i) {
                const auto& e = edgelist[i];
                if (e.x >= 0 && e.x < _shape[1] && e.y >= 0 && e.y < _shape[0]) {
                    r(i) = BilinearSample<float>(_DF, {e.x, e.y}) / edgelist.size();
                    v(i) = 1.0;
                    X.row(i) = Rp.transpose() * _Kinv * Vec3{e.x, e.y, 1.0} * e.depth 
                        - Rp.transpose() * Tp;
                } else {
                    v(i) = 0.0;
                }
            }
            return std::make_tuple(r, v);
        } else {
            r.resize(X.rows());
            r.setZero();

            v.resize(X.rows());
            v.setZero();

            for (int i = 0; i < X.rows(); ++i) {
                Vec3 Xc = _K * (Rp * Vec3{X.row(i)} + Tp);
                Vec2 x = Xc.head<2>() / Xc(2);
                if (anynan(x)) {
                    v(i) = 0.0;
                } else {
                    if (x(0) >= 0 && x(0) < _shape[1] && x(1) >= 0 && x(1) < _shape[0]) {
                        r(i) = BilinearSample<float>(_DF, x) / X.rows();
                        v(i) = 1.0;
                    } else {
                        v(i) = 0.0;
                    }
                }
            }
            return std::make_tuple(r, v);
        }
    }

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
    Mat3 _K, _Kinv;
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
    float Tnoise = 0.2;
    float Rnoise = 0.5;

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
    cv::namedWindow("depth", CV_WINDOW_NORMAL);

    feh::Mat3 flip;
    flip << -1, 0, 0,
         0, -1, 0,
         0, 0, 1;

    for (int i = 0; i < 1000; ++i) {
        std::cout << "==========\n";
        auto cost = tracker.Minimize(1);
        std::cout << "Iter=" << i << std::endl;
        std::cout << "Cost=" << cost << std::endl;

        feh::Mat3 Re;
        feh::Vec3 Te;
        std::tie(Re, Te) = tracker.GetEstimate();
        Re = flip * Re;
        Te = flip * Te; // flip back
        Eigen::Matrix<float, 6, 1> err;
        err << invrodrigues(feh::Mat3{Re.transpose() * packet._go.so3().matrix()}),
            Te - packet._go.translation();
        std::cout << "Error vector=" << err.transpose() << std::endl;
        std::cout << "R Error=" << err.head<3>().norm() / 3.14 * 180 << std::endl;
        std::cout << "T Error=" << 100 * err.tail<3>().norm() / (packet._V.maxCoeff() - packet._V.minCoeff())<< std::endl;

        auto depth = tracker.RenderEstimate();
        cv::imshow("depth", depth);
        cv::waitKey(10);
    }

}

