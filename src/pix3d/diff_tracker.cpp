#include "pix3d/diff_tracker.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"


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
    _R(R), _T(T), _V(V), _F(F),
    _timer("diff_tracker")
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
    _DF = cv::Mat(_shape[0], _shape[1], CV_32FC1);
    DistanceTransform::BuildView(tmp).convertTo(_DF, CV_32FC1);    // DF value range [0, 1]
    _DF /= 255.0;
    cv::exp(_DF / 255.0, _DF);
    cv::GaussianBlur(_DF, _DF, cv::Size(3, 3), 0, 0);

    cv::Scharr(_DF, _dDF_dx, CV_32FC1, 1, 0, 3);
    cv::Scharr(_DF, _dDF_dy, CV_32FC1, 0, 1, 3);
}

ftype DiffTracker::Minimize(int steps=1) {
    VecX r, rp;     // current and predicted residual
    MatX J, Jp;     // current and predicted Jacobian
    ftype cost, costp;  // current cost and predicted cost
    // predicted rotation & translation
    Mat3 Rp; 
    Vec3 Tp;
    for (int iter = 0; iter < steps; ++iter) {
        _timer.Tick("Jacobian");
        std::tie(r, J) = ComputeResidualAndJacobian(_R, _T);
        // std::cout << folly::sformat("J.shape=({},{})", J.rows(), J.cols()) << std::endl;
        cost = 0.5 * r.squaredNorm();
        _timer.Tock("Jacobian");

        _timer.Tick("GNupdate");
        // Gauss-Newton update
        MatX JtJ = J.transpose() * J;
        // ftype damping = 0;
        // JtJ.diagonal() *= (1+damping);
        Eigen::Matrix<ftype, 6, 1> delta = -JtJ.ldlt().solve(J.transpose() * r);

#ifdef PIX3D_LINE_SEARCH
        ftype dcost = r.transpose() * J * delta;
        std::cout << "dcost=" << dcost << std::endl;
        ftype alpha = 0.1, beta = 0.9, stepsize = 10;
        ftype best_stepsize = 1;
        int linesearch_step;
        for (linesearch_step = 0; linesearch_step < 10; ++linesearch_step) {
            Rp = _R * rodrigues(Vec3{stepsize * delta.head<3>()});
            Tp = _T + stepsize * delta.tail<3>();
            VecX rp;
            std::tie(rp, std::ignore) = ComputeResidualAndJacobian(Rp, Tp);
            costp = 0.5 * rp.squaredNorm();
            if (costp < cost) {
                costp = cost;
                best_stepsize = stepsize;
            } else {
                stepsize *= beta;
            } 
        }
        std::cout << "linesearch_step=" << linesearch_step << ";;;stepsize=" << stepsize << std::endl;
        _R = _R * rodrigues(Vec3{best_stepsize * delta.head<3>()});
        _T = _T + best_stepsize * delta.tail<3>();
#else 
        _R = _R * rodrigues(Vec3{delta.head<3>()});
        _T = _T + delta.tail<3>();
#endif
        _timer.Tock("GNupdate");
    }
    std::cout << _timer;

    return cost;
}


std::tuple<VecX, MatX> DiffTracker::ComputeResidualAndJacobian(
        const Mat3 &R, const Vec3 &T) {
    VecX r, v;  // residual and valid bit
    MatX J;

    // no 3D points yet
    _timer.Tick("Transform");
    auto V = TransformShape(R, T);
    _timer.Tock("Transform");

    _timer.Tick("SetMesh");
    _engine->SetMesh((float*)V.data(), V.rows(), (int*)_F.data(), _F.rows());
    _timer.Tock("SetMesh");

    _timer.Tick("Render");
    Eigen::Matrix<ftype, 4, 4, Eigen::ColMajor> identity;
    identity.setIdentity();
    std::vector<EdgePixel> edgelist;
    _engine->ComputeEdgePixels(identity, edgelist);
    _timer.Tock("Render");

    r.resize(edgelist.size());
    r.setZero();

    J.resize(edgelist.size(), 6);
    J.setZero();

    v.resize(edgelist.size());
    v.setZero();

    auto fill_jacobian_kernel = [&edgelist, &r, &v, &J, &R, &T, this](const tbb::blocked_range<int> &range) {
        for (int i = range.begin(); i < range.end(); ++i) {
            const auto& e = edgelist[i];
            if (e.x >= 0 && e.x < this->_shape[1] && e.y >= 0 && e.y < this->_shape[0]) {
                r(i) = BilinearSample<float>(this->_DF, {e.x, e.y}); // / edgelist.size();
                v(i) = 1.0;
                // back-project to object frame
                Vec3 Xc = this->_Kinv * Vec3{e.x, e.y, 1.0} * e.depth;
                Vec3 Xo = R.transpose() * (Xc - T);
                // measurement equation: DF(x) = DF(\pi( R Xo + T))
                Eigen::Matrix<ftype, 3, 6> dXc_dwt;
                dXc_dwt << dAB_dA(Mat3{}, Xo) * dAB_dB(R, Mat3{}) * dhat(Vec3{}), Mat3::Identity();

                Eigen::Matrix<ftype, 2, 3> dx_dXc;
                ftype Zinv = 1.0 / Xc(2);
                ftype Zinv2 = Zinv * Zinv;
                dx_dXc << this->_K(0, 0) * Zinv, 0, -this->_K(0, 0) * Xc(0) * Zinv2,
                      0, this->_K(1, 1) * Zinv, -this->_K(1, 1) * Xc(1) * Zinv2;

                Eigen::Matrix<ftype, 2, 6> dx_dwt{dx_dXc * dXc_dwt};

                Eigen::Matrix<ftype, 1, 2> dDF_dx{
                    this->_dDF_dx.at<float>((int)e.y, (int)e.x),
                    this->_dDF_dy.at<float>((int)e.y, (int)e.x)};
                J.row(i) = dDF_dx * dx_dwt;
                // J.row(i) /= edgelist.size();
            } else {
                v(i) = 0.0;
            }
        }
    };
    _timer.Tick("FillJacobian");
    tbb::parallel_for(tbb::blocked_range<int>(0, edgelist.size()),
            fill_jacobian_kernel,
            tbb::auto_partitioner());
    _timer.Tock("FillJacobian");

    return std::make_tuple(r, J);
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
