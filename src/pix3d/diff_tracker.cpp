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
    img_(img.clone()), edge_(edge.clone()),
    shape_(shape), R_(R), T_(T), V_(V), F_(F), timer_("diff_tracker")
{
    K_ << fx, 0, cx,
    0, fy, cy,
    0, 0, 1;
    Kinv_ = K_.inverse();

    engine_ = std::make_shared<Renderer>(shape_[0], shape_[1]);
    engine_->SetCamera(znear, zfar, fx, fy, cx, cy);
    Mat3 flip;
    flip << -1, 0, 0,
        0, -1, 0,
        0, 0, 1;
    R_ = flip * R_;
    T_ = flip * T_;

    BuildDistanceField();
}


void DiffTracker::BuildDistanceField() {
    cv::Mat normalized_edge, tmp;
    normalized_edge = cv::Scalar::all(255) - edge_;
    cv::distanceTransform(normalized_edge / 255.0, tmp, CV_DIST_L2, CV_DIST_MASK_PRECISE);
    DF_ = cv::Mat(shape_[0], shape_[1], CV_32FC1);
    DistanceTransform::BuildView(tmp).convertTo(DF_, CV_32FC1);    // DF value range [0, 1]
    // DF_ /= 255.0;
    cv::exp(DF_ / 255.0 - 2.0, DF_);
    cv::GaussianBlur(DF_, DF_, cv::Size(3, 3), 0, 0);

    cv::Scharr(DF_, dDF_dx_, CV_32FC1, 1, 0, 3);
    cv::Scharr(DF_, dDF_dy_, CV_32FC1, 0, 1, 3);
}


ftype DiffTracker::Minimize(int steps=1) {
    VecX r, rp;     // current and predicted residual
    MatX J, Jp;     // current and predicted Jacobian
    ftype cost = -1, costp;  // current cost and predicted cost
    // predicted rotation & translation
    Mat3 Rp; 
    Vec3 Tp;
    for (int iter = 0; iter < steps; ++iter) {
        timer_.Tick("Jacobian");
        std::tie(r, J) = ComputeResidualAndJacobian(R_, T_);
        // std::cout << folly::sformat("J.shape=({},{})", J.rows(), J.cols()) << std::endl;
        cost = 0.5 * r.squaredNorm();
        timer_.Tock("Jacobian");

        timer_.Tick("GNupdate");
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
            Rp = R_ * rodrigues(Vec3{stepsize * delta.head<3>()});
            Tp = T_ + stepsize * delta.tail<3>();
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
        R_ = R_ * rodrigues(Vec3{best_stepsize * delta.head<3>()});
        T_ = T_ + best_stepsize * delta.tail<3>();
#else 
        R_ = R_ * rodrigues(Vec3{delta.head<3>()});
        T_ = T_ + delta.tail<3>();
#endif
        timer_.Tock("GNupdate");
    }
//    std::cout << timer_;

    return cost;
}


std::tuple<VecX, MatX> DiffTracker::ComputeResidualAndJacobian(
        const Mat3 &R, const Vec3 &T) {
    VecX r, v;  // residual and valid bit
    MatX J;

    // no 3D points yet
    timer_.Tick("Transform");
    auto V = TransformShape(R, T);
    timer_.Tock("Transform");

    timer_.Tick("SetMesh");
    engine_->SetMesh(V, F_);
    timer_.Tock("SetMesh");

    timer_.Tick("Render");
    std::vector<EdgePixel> edgelist;
    engine_->ComputeEdgePixels(Mat4f::Identity(), edgelist);
    timer_.Tock("Render");

    r.resize(edgelist.size());
    r.setZero();

    J.resize(edgelist.size(), 6);
    J.setZero();

    v.resize(edgelist.size());
    v.setZero();

    auto fill_jacobian_kernel = [&edgelist, &r, &v, &J, &R, &T, this](const tbb::blocked_range<int> &range) {
        for (int i = range.begin(); i < range.end(); ++i) {
            const auto& e = edgelist[i];
            if (e.x >= 0 && e.x < this->shape_[1] && e.y >= 0 && e.y < this->shape_[0]) {
                r(i) = BilinearSample<float>(this->DF_, {e.x, e.y}) / edgelist.size();
                v(i) = 1.0;
                // back-project to object frame
                Vec3 Xc = this->Kinv_ * Vec3{e.x, e.y, 1.0} * e.depth;
                Vec3 Xo = R.transpose() * (Xc - T);
                // measurement equation: DF(x) = DF(\pi( R Xo + T))
                Eigen::Matrix<ftype, 3, 6> dXc_dwt;
                dXc_dwt << dAB_dA(Mat3{}, Xo) * dAB_dB(R, Mat3{}) * dhat(Vec3{}), Mat3::Identity();

                Eigen::Matrix<ftype, 2, 3> dx_dXc;
                ftype Zinv = 1.0 / Xc(2);
                ftype Zinv2 = Zinv * Zinv;
                dx_dXc << this->K_(0, 0) * Zinv, 0, -this->K_(0, 0) * Xc(0) * Zinv2,
                      0, this->K_(1, 1) * Zinv, -this->K_(1, 1) * Xc(1) * Zinv2;

                Eigen::Matrix<ftype, 2, 6> dx_dwt{dx_dXc * dXc_dwt};

                Eigen::Matrix<ftype, 1, 2> dDF_dx{
                    this->dDF_dx_.at<float>((int)e.y, (int)e.x),
                    this->dDF_dy_.at<float>((int)e.y, (int)e.x)};
                J.row(i) = dDF_dx * dx_dwt;
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


cv::Mat DiffTracker::RenderEstimate() const {
    auto V = TransformShape(R_, T_);
    engine_->SetMesh(V, F_);
    cv::Mat depth(shape_[0], shape_[1], CV_32FC1);
    engine_->RenderDepth(Mat4::Identity(), depth);
    return depth;
}


cv::Mat DiffTracker::RenderEdgepixels() const {
    std::vector<EdgePixel> edgelist;
    engine_->ComputeEdgePixels(Mat4::Identity(), edgelist);

    cv::Mat out(img_.clone());
    for (const auto &e : edgelist) {
        if (e.x >= 0 && e.x < shape_[1] && e.y >= 0 && e.y < shape_[0]) {
            cv::circle(out, cv::Point((int)e.x, (int)e.y), 2, cv::Scalar(255, 0, 0), -1);
        }
    }
    return out;
}


}   // namespace feh
