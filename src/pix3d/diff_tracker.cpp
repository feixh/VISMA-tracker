#include "pix3d/diff_tracker.h"

constexpr float znear = 0.1;
constexpr float zfar = 10;

namespace feh {

DiffTracker::DiffTracker(const cv::Mat &img, const cv::Mat &edge,
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


    cv::Mat tmp, normalized_edge;
    _edge.convertTo(tmp, CV_32FC1);
    tmp = (255.0 - tmp) * 100;
    cv::GaussianBlur(tmp, tmp, cv::Size(3, 3), 0, 0);
    // cv::Mat index(_shape[0], _shape[1], CV_32SC2);
    DistanceTransform{}(tmp, _DF);
    DistanceTransform::BuildView(_DF).convertTo(_DF, CV_32FC1);

    cv::Scharr(_DF, _dDF_dx, CV_32FC1, 1, 0, 3);
    cv::Scharr(_DF, _dDF_dy, CV_32FC1, 0, 1, 3);
    
   
    // _dDF_dx = cv::Mat(_shape[0], _shape[1], CV_32FC1);
    // _dDF_dy = cv::Mat(_shape[0], _shape[1], CV_32FC1);
    // for (int i = 0; i < _shape[0]; ++i) {
    //     for (int j = 0; j < _shape[1]; ++j) {
    //         auto target = index.at<cv::Vec2i>(i, j);
    //         Vec2 grad{target(0)-i, target(1)-j};
    //         if (grad.norm() > 1e-8) {
    //             grad /= grad.norm();
    //         }
    //         _dDF_dx.at<float>(i, j) = grad(1);
    //         _dDF_dy.at<float>(i, j) = grad(0);
    //     }
    // }
}

ftype DiffTracker::Minimize(int steps=1) {
    ftype stepsize = 1e-1;
    VecX r;
    MatX J;
    for (int iter = 0; iter < steps; ++iter) {
        std::tie(r, J) = ComputeLoss2();
        // std::cout << folly::sformat("J.shape=({},{})", J.rows(), J.cols()) << std::endl;

        // Gauss-Newton update
        MatX JtJ = J.transpose() * J;
        MatX damping(JtJ.rows(), JtJ.cols());
        damping.setIdentity();
        damping *= 0;
        Eigen::Matrix<ftype, 6, 1> delta = -stepsize * (JtJ + damping).ldlt().solve(J.transpose() * r);
        // _R = _R + hat<ftype>(delta.head<3>());
        _R = _R * rodrigues(Vec3{delta.head<3>()});
        _T = _T + delta.tail<3>();

        Eigen::JacobiSVD<MatX> svd(JtJ);
        std::cout << "SingularValues=" << svd.singularValues().transpose() << std::endl;
    }
    return r.sum();
}

std::tuple<VecX, MatX> DiffTracker::ComputeLoss() const {
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


std::tuple<VecX, MatX> DiffTracker::ComputeLoss2() const {
    VecX r, v;  // residual and valid bit
    MatX J;

    // no 3D points yet
    auto V = TransformShape(_R, _T);
    _engine->SetMesh((float*)V.data(), V.rows(), (int*)_F.data(), _F.rows());
    Eigen::Matrix<ftype, 4, 4, Eigen::ColMajor> identity;
    identity.setIdentity();
    std::vector<EdgePixel> edgelist;
    _engine->ComputeEdgePixels(identity, edgelist);

    r.resize(edgelist.size());
    r.setZero();

    J.resize(edgelist.size(), 6);
    J.setZero();

    v.resize(edgelist.size());
    v.setZero();

    ftype cost = 0;
    for (int i = 0; i < edgelist.size(); ++i) {
        const auto& e = edgelist[i];
        if (e.x >= 0 && e.x < _shape[1] && e.y >= 0 && e.y < _shape[0]) {
            // std::cout << folly::sformat("{:04d}:(x,y,z)=({}, {}, {})\n", i, e.x, e.y, e.depth);
            // r(i) = BilinearSample<float>(_DF, {e.x, e.y}) / edgelist.size();
            r(i) = BilinearSample<float>(_DF, {e.x, e.y});
            v(i) = 1.0;
            // back-project to object frame
            Vec3 Xc = _Kinv * Vec3{e.x, e.y, 1.0} * e.depth;
            Vec3 Xo = _R.transpose() * (Xc - _T);
            // measurement equation: DF(x) = DF(\pi( R Xo + T))
            Eigen::Matrix<ftype, 3, 6> dXc_dwt;
            dXc_dwt << dAB_dA(Mat3{}, Xo) * dhat(Vec3{}), Mat3::Identity();

            Eigen::Matrix<ftype, 2, 3> dx_dXc;
            dx_dXc << _K(0, 0) / Xc(2), 0, -_K(0, 0) * Xc(0)  / (Xc(2) * Xc(2)),
                  0, _K(1, 1) / Xc(2), -_K(1, 1) * Xc(1)  / (Xc(2) * Xc(2));

            Eigen::Matrix<ftype, 2, 6> dx_dwt{dx_dXc * dXc_dwt};

            Eigen::Matrix<ftype, 1, 2> dDF_dx{
                _dDF_dx.at<float>((int)e.y, (int)e.x),
                _dDF_dy.at<float>((int)e.y, (int)e.x)};
#ifdef PIX3D_VERBOSE
            std::cout << "r=" << r(i) << std::endl;
            std::cout << "dXc_dwt=\n" << dXc_dwt << std::endl;
            std::cout << "dx_dXc=\n" << dx_dXc << std::endl;
            std::cout << "dDF_dx=\n" << dDF_dx << std::endl;
#endif

            J.row(i) = dDF_dx * dx_dwt;
#ifdef PIX3D_VERBOSE
            std::cout << J.row(i) << std::endl;
#endif
            // J.row(i) /= edgelist.size();

        } else {
            v(i) = 0.0;
        }
    }
    return std::make_tuple(r, J);
}

/// \brief: Compute the loss at the current pose with given perturbation
std::tuple<VecX, VecX> DiffTracker::ForwardPass(
        const Vec3 &dW, const Vec3 &dT, 
        Eigen::Matrix<ftype, Eigen::Dynamic, 3> &X) const {

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
                X.row(i) = Rp.transpose() * (_Kinv * Vec3{e.x, e.y, 1.0} * e.depth - Tp);
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
                    // r(i) = std::sqrt(BilinearSample<float>(_DF, x)) / X.rows();
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

cv::Mat DiffTracker::RenderEstimate() const {
    auto V = TransformShape(_R, _T);
    _engine->SetMesh((float*)V.data(), V.rows(), (int*)_F.data(), _F.rows());
    Eigen::Matrix<ftype, 4, 4, Eigen::ColMajor> identity;
    identity.setIdentity();
    cv::Mat depth(_shape[0], _shape[1], CV_32FC1); 
    _engine->RenderDepth(identity, depth);
    return depth;
}


cv::Mat DiffTracker::RenderEdgepixels() const {
    Eigen::Matrix<ftype, 4, 4, Eigen::ColMajor> identity;
    identity.setIdentity();
    std::vector<EdgePixel> edgelist;
    _engine->ComputeEdgePixels(identity, edgelist);

    cv::Mat out(_img.clone());
    for (const auto &e : edgelist) {
        if (e.x >= 0 && e.x < _shape[1] && e.y >= 0 && e.y < _shape[0]) {
            cv::circle(out, cv::Point((int)e.x, (int)e.y), 2, cv::Scalar(255, 0, 0), -1);
        }
    }
    return out;

}


}   // namespace feh
