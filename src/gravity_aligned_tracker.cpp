#include "gravity_aligned_tracker.h"


constexpr float znear = 0.1;
constexpr float zfar = 10;

namespace feh {


ftype GravityAlignedTracker::Minimize(int steps=1) {
    VecX r, rp;     // current and predicted residual
    MatX J, Jp;     // current and predicted Jacobian
    ftype cost = -1, costp;  // current cost and predicted cost
    ftype stepsize = 0.1;
    // predicted rotation & translation
    SE3 gp;
    for (int iter = 0; iter < steps; ++iter) {
        timer_.Tick("Jacobian");
        std::tie(r, J) = ComputeResidualAndJacobian(g_);    // J: N x 4
        // std::cout << folly::sformat("J.shape=({},{})", J.rows(), J.cols()) << std::endl;
        cost = 0.5 * r.squaredNorm();
        timer_.Tock("Jacobian");

        timer_.Tick("GNupdate");
        // Gauss-Newton update
        MatX JtJ = J.transpose() * J;
        Eigen::Matrix<ftype, 4, 1> delta = -stepsize * JtJ.ldlt().solve(J.transpose() * r);
        g_.so3() = g_.so3() * SO3::exp(delta(0) * gamma_);
        g_.translation() += delta.tail<3>();
        timer_.Tock("GNupdate");
    }
//    std::cout << timer_;

    return cost;
}

std::tuple<VecX, MatX> GravityAlignedTracker::ComputeResidualAndJacobian(const SE3 &g) {
    VecX r;
    MatX J, Jtmp;
    // re-use the parent's jacobian computation
    std::tie(r, Jtmp) = DFTracker::ComputeResidualAndJacobian(g);
    J.resize(Jtmp.rows(), 4);
    J << Jtmp.leftCols(3) * gamma_, Jtmp.rightCols(3);
    return std::make_tuple(r, J);
}

void GravityAlignedTracker::UpdateGravity(const SO3 &Rg) {
    Rg_ = Rg;
    gamma_ = Rg_ * Vec3{0, 1, 0};
    gamma_ /= gamma_.norm();
}


}   // namespace feh
