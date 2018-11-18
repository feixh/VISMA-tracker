#include "gravity_aligned_tracker.h"


constexpr float znear = 0.1;
constexpr float zfar = 10;

namespace feh {

DFTracker::DFTracker(const cv::Mat &img, const cv::Mat &edge,
            const Vec2i &shape, 
            ftype fx, ftype fy, ftype cx, ftype cy,
            const SE3 &g,
            const MatX &V, const MatXi &F):
    img_(img.clone()), edge_(edge.clone()),
    shape_(shape), g_(g), V_(V), F_(F), timer_("diff_tracker")
{
    K_ << fx, 0, cx,
    0, fy, cy,
    0, 0, 1;
    Kinv_ = K_.inverse();

    engine_ = std::make_shared<Renderer>(shape_[0], shape_[1]);
    engine_->SetCamera(znear, zfar, fx, fy, cx, cy);
    engine_->SetMesh(V_, F_);


    Mat3 flip_co;   // object -> camera
    flip_co << -1, 0, 0,
        0, -1, 0,
        0, 0, 1;
    Mat3 flip_sc;   // camera -> spatial
    flip_sc << 0, 0, 1,
               -1, 0, 0,
               0, -1, 0;
    Mat3 flip_so{flip_sc * flip_co};   // object -> spatial
    g_ = SE3(SO3{flip_so * g.so3().matrix()}, flip_so * g.translation());

    BuildDistanceField();
}


void DFTracker::BuildDistanceField() {
    cv::Mat normalized_edge, tmp;
    normalized_edge = cv::Scalar::all(255) - edge_;
    cv::distanceTransform(normalized_edge / 255.0, tmp, CV_DIST_L2, CV_DIST_MASK_PRECISE);
    DF_ = cv::Mat(shape_[0], shape_[1], CV_32FC1);
    DistanceTransform::BuildView(tmp).convertTo(DF_, CV_32FC1);    // DF value range [0, 1]
    // DF_ /= 255.0;
    cv::exp(DF_ / 255.0 - 2.0, DF_);
    cv::GaussianBlur(DF_, DF_, cv::Size(3, 3), 0, 0);

    cv::Mat dDF_dx, dDF_dy;
    cv::Scharr(DF_, dDF_dx, CV_32FC1, 1, 0, 3);
    cv::Scharr(DF_, dDF_dy, CV_32FC1, 0, 1, 3);
    cv::merge(std::vector<cv::Mat>{dDF_dx, dDF_dy}, dDF_dxy_);
}


ftype DFTracker::Minimize(int steps=1) {
    VecX r, rp;     // current and predicted residual
    MatX J, Jp;     // current and predicted Jacobian
    ftype cost = -1, costp;  // current cost and predicted cost
    // predicted rotation & translation
    SE3 gp;
    for (int iter = 0; iter < steps; ++iter) {
        timer_.Tick("Jacobian");
        std::tie(r, J) = ComputeResidualAndJacobian(g_);
        cost = 0.5 * r.squaredNorm();
        timer_.Tock("Jacobian");

        timer_.Tick("GNupdate");
        // Gauss-Newton update
        MatX JtJ = J.transpose() * J;
        Eigen::Matrix<ftype, 6, 1> delta = -JtJ.ldlt().solve(J.transpose() * r);
        g_.so3() = g_.so3() * SO3::exp(delta.head<3>());
        // g_.so3() = g_.so3() * SO3::exp(Vec3f{0, delta(1), 0});
        g_.translation() += delta.tail<3>();
        timer_.Tock("GNupdate");
    }
//    std::cout << timer_;

    return cost;
}


std::tuple<VecX, MatX> DFTracker::ComputeResidualAndJacobian(const SE3 &g) {
    // Measurement process:
    // x = Proj(Xc) = Proj(Rcs * Xs + Tcs) + Proj(Rcs * (Rso * Xo + Tso) + Tcs)
    // [Rcs | Tcs]: spatial to camera transformation
    // [Rso | Tso]: object to spatial transformation
    // Xs: 3D point in spatial frame
    // Xo: 3D point in object frame
    // Xc: 3D point in camera frame
    // x: projection of the 3D point
    
    VecX r, v;  // residual and valid bit
    MatX J;

    // no 3D points yet
    timer_.Tick("Render");
    std::vector<EdgePixel> edgelist;
    engine_->ComputeEdgePixels(g.matrix(), edgelist);
    timer_.Tock("Render");

    r.resize(edgelist.size());
    r.setZero();

    J.resize(edgelist.size(), 6);
    J.setZero();

    v.resize(edgelist.size());
    v.setZero();

    auto Rso = g.so3().matrix();
    auto Tso = g.translation();
    auto Rcs = gsc_.inv().so3().matrix();
    auto Tcs = gsc_.inv().translation();

    auto fill_jacobian_kernel = [&edgelist, &r, &v, &J, &Rso, &Tso, &Rcs, &Tcs, this]
        (const tbb::blocked_range<int> &range) 
        {
        for (int i = range.begin(); i < range.end(); ++i) {
            const auto& e = edgelist[i];
            if (e.x >= 0 && e.x < this->shape_[1] && e.y >= 0 && e.y < this->shape_[0]) {
                r(i) = BilinearSample<float>(this->DF_, {e.x, e.y}) / edgelist.size();
                v(i) = 1.0;
                // back-project to object frame
                Vec3 Xc = this->Kinv_ * Vec3{e.x, e.y, 1.0} * e.depth;

                Vec3 Xs = Rcs.transpose() * (Xc - Tcs);
                Mat3 dXc_dXs = Rcs;     // Xc = Rcs * Xs + Tcs

                Vec3 Xo = Rso.transpose() * (Xs - Tso);

                // Xs = Rso(1+\hat w) Xo + Tso
                Mat3 dXs_dt = Mat3::Identity();
                Mat3 dXs_dw = dAB_dA(Mat3{}, Xo) * dAB_dB(Rso, Mat3{}) * dhat(Vec3{});  
                Eigen::Matrix<ftype, 3, 6> dXs_dwt;
                dXs_dwt << dXs_dw, dXs_dt;

                // measurement equation: DF(x) = DF(\pi( R Xo + T))
                Eigen::Matrix<ftype, 3, 6> dXc_dwt;
                dXc_dwt << dXc_dXs * dXs_dwt;

                Eigen::Matrix<ftype, 2, 3> dx_dXc;
                ftype Zinv = 1.0 / Xc(2);
                ftype Zinv2 = Zinv * Zinv;
                dx_dXc << this->K_(0, 0) * Zinv, 0, -this->K_(0, 0) * Xc(0) * Zinv2,
                      0, this->K_(1, 1) * Zinv, -this->K_(1, 1) * Xc(1) * Zinv2;

                Eigen::Matrix<ftype, 2, 6> dx_dwt{dx_dXc * dXc_dwt};

                auto dDF_dx = this->dDF_dxy_.at<cv::Vec2f>((int)e.y, (int)e.x);
                J.row(i) = Eigen::Matrix<ftype, 1, 2>{dDF_dx(0), dDF_dx(1)} * dx_dwt;
                J.row(i) /= edgelist.size();
            } else {
                v(i) = 0.0;
            }
        }
    };
    timer_.Tick("FillJacobian");
    tbb::parallel_for(tbb::blocked_range<int>(0, edgelist.size()),
            fill_jacobian_kernel,
            tbb::auto_partitioner());
    timer_.Tock("FillJacobian");

    return std::make_tuple(r, J);
}


cv::Mat DFTracker::RenderEstimate() const {
    cv::Mat depth(shape_[0], shape_[1], CV_32FC1);
    engine_->RenderDepth(g_.matrix(), depth);
    return depth;
}


cv::Mat DFTracker::RenderEdgepixels() const {
    std::vector<EdgePixel> edgelist;
    engine_->ComputeEdgePixels(g_.matrix(), edgelist);

    cv::Mat out(img_.clone());
    for (const auto &e : edgelist) {
        if (e.x >= 0 && e.x < shape_[1] && e.y >= 0 && e.y < shape_[0]) {
            cv::circle(out, cv::Point((int)e.x, (int)e.y), 2, cv::Scalar(255, 0, 0), -1);
        }
    }
    return out;
}

void DFTracker::UpdateCameraPose(const SE3 &gsc) {
    gsc_ = gsc;
    // Note: rendering engine takes transformation from initial frame (spatial) to current frame (camera)
    engine_->SetCamera(gsc_.inv().matrix());
}

void DFTracker::UpdateImage(const cv::Mat &img, const cv::Mat &edge) {
        img_ = img.clone();
        edge_ = edge.clone();
        BuildDistanceField();
}


}   // namespace feh
