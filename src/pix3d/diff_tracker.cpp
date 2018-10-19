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


    // cv::Mat tmp, normalized_edge;
    // _edge.convertTo(tmp, CV_32FC1);
    // tmp = (255.0 - tmp) / 255.0;
    // // cv::GaussianBlur(tmp, tmp, cv::Size(3, 3), 0, 0);
    // // // cv::Mat index(_shape[0], _shape[1], CV_32SC2);
    // DistanceTransform{}(tmp, _DF);

    cv::Mat normalized_edge, tmp;
    normalized_edge = cv::Scalar::all(255) - _edge;
    cv::distanceTransform(normalized_edge / 255.0, tmp, CV_DIST_L2, CV_DIST_MASK_PRECISE);
    _DF = cv::Mat(_shape[0], _shape[1], CV_32FC1);
    DistanceTransform::BuildView(tmp).convertTo(_DF, CV_32FC1);    // DF value range [0, 1]
    _DF /= 255.0;
    cv::GaussianBlur(_DF, _DF, cv::Size(3, 3), 0, 0);

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
    ftype stepsize = 1;
    VecX r;
    MatX J;
    for (int iter = 0; iter < steps; ++iter) {
        _timer.Tick("Jacobian");
        std::tie(r, J) = ComputeResidualAndJacobian();
        // std::cout << folly::sformat("J.shape=({},{})", J.rows(), J.cols()) << std::endl;
        _timer.Tock("Jacobian");

        _timer.Tick("GNupdate");
        // Gauss-Newton update
        MatX JtJ = J.transpose() * J;
        ftype damping = 1e-3;
        JtJ.diagonal() *= (1+damping);
        Eigen::Matrix<ftype, 6, 1> delta = -JtJ.ldlt().solve(J.transpose() * r);
        // _R = _R + _R * hat<ftype>(delta.head<3>());
        _R = _R * rodrigues(Vec3{delta.head<3>()});
        _T = _T + delta.tail<3>();
        _timer.Tock("GNupdate");

#ifdef PIX3D_VERBOSE
        Eigen::JacobiSVD<MatX> svd(JtJ);
        std::cout << "SingularValues=" << svd.singularValues().transpose() << std::endl;
#endif
    }
    std::cout << _timer;
    return r.sum();
}


std::tuple<VecX, MatX> DiffTracker::ComputeResidualAndJacobian() {
    VecX r, v;  // residual and valid bit
    MatX J;

    // no 3D points yet
    _timer.Tick("Transform");
    auto V = TransformShape(_R, _T);
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


#ifdef PIX3D_JACOBIAN_PARALLEL_FILLIN
    auto fill_jacobian_kernel = [&edgelist, &r, &v, &J, this](const tbb::blocked_range<int> &range) {
        for (int i = range.begin(); i < range.end(); ++i) {
            const auto& e = edgelist[i];
            if (e.x >= 0 && e.x < this->_shape[1] && e.y >= 0 && e.y < this->_shape[0]) {
                // std::cout << folly::sformat("{:04d}:(x,y,z)=({}, {}, {})\n", i, e.x, e.y, e.depth);
                r(i) = BilinearSample<float>(this->_DF, {e.x, e.y}); // / edgelist.size();
                // r(i) = _DF.at<float>((int)e.y, (int)e.x);
                v(i) = 1.0;
                // back-project to object frame
                Vec3 Xc = this->_Kinv * Vec3{e.x, e.y, 1.0} * e.depth;
                Vec3 Xo = this->_R.transpose() * (Xc - this->_T);
                // measurement equation: DF(x) = DF(\pi( R Xo + T))
                Eigen::Matrix<ftype, 3, 6> dXc_dwt;
                dXc_dwt << dAB_dA(Mat3{}, Xo) * dAB_dB(this->_R, Mat3{}) * dhat(Vec3{}), Mat3::Identity();

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
                
#ifdef PIX3D_VERBOSE
                std::cout << "r=" << r(i) << std::endl;
                std::cout << "dXc_dwt=\n" << dXc_dwt << std::endl;
                std::cout << "dx_dXc=\n" << dx_dXc << std::endl;
                std::cout << "dDF_dx=\n" << dDF_dx << std::endl;
                std::cout << J.row(i) << std::endl;
#endif
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
#endif

    _timer.Tick("FillJacobian");
    for (int i = 0; i < edgelist.size(); ++i) {
        const auto& e = edgelist[i];
        if (e.x >= 0 && e.x < _shape[1] && e.y >= 0 && e.y < _shape[0]) {
            // std::cout << folly::sformat("{:04d}:(x,y,z)=({}, {}, {})\n", i, e.x, e.y, e.depth);
            r(i) = BilinearSample<float>(_DF, {e.x, e.y}); // / edgelist.size();
            // r(i) = _DF.at<float>((int)e.y, (int)e.x);
            v(i) = 1.0;
            // back-project to object frame
            Vec3 Xc = _Kinv * Vec3{e.x, e.y, 1.0} * e.depth;
            Vec3 Xo = _R.transpose() * (Xc - _T);
            // measurement equation: DF(x) = DF(\pi( R Xo + T))
            Eigen::Matrix<ftype, 3, 6> dXc_dwt;
            dXc_dwt << dAB_dA(Mat3{}, Xo) * dAB_dB(_R, Mat3{}) * dhat(Vec3{}), Mat3::Identity();

            Eigen::Matrix<ftype, 2, 3> dx_dXc;
            ftype Zinv = 1.0 / Xc(2);
            ftype Zinv2 = Zinv * Zinv;
            dx_dXc << _K(0, 0) * Zinv, 0, -_K(0, 0) * Xc(0) * Zinv2,
                  0, _K(1, 1) * Zinv, -_K(1, 1) * Xc(1) * Zinv2;

            Eigen::Matrix<ftype, 2, 6> dx_dwt{dx_dXc * dXc_dwt};

            Eigen::Matrix<ftype, 1, 2> dDF_dx{
                _dDF_dx.at<float>((int)e.y, (int)e.x),
                _dDF_dy.at<float>((int)e.y, (int)e.x)};
            J.row(i) = dDF_dx * dx_dwt;
            // J.row(i) /= edgelist.size();
            
#ifdef PIX3D_VERBOSE
            std::cout << "r=" << r(i) << std::endl;
            std::cout << "dXc_dwt=\n" << dXc_dwt << std::endl;
            std::cout << "dx_dXc=\n" << dx_dXc << std::endl;
            std::cout << "dDF_dx=\n" << dDF_dx << std::endl;
            std::cout << J.row(i) << std::endl;
#endif
        } else {
            v(i) = 0.0;
        }
    }
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
