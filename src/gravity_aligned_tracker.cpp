#include "gravity_aligned_tracker.h"


constexpr float znear = 0.1;
constexpr float zfar = 10;

namespace feh {

// GravityAlignedTracker::GravityAlignedTracker(const cv::Mat &img, const cv::Mat &edge,
//             const Vec2i &shape, 
//             ftype fx, ftype fy, ftype cx, ftype cy,
//             const SE3 &g,
//             const MatX &V, const MatXi &F):
//     img_(img.clone()), edge_(edge.clone()),
//     shape_(shape), g_(g), V_(V), F_(F), timer_("diff_tracker")
// {
//     K_ << fx, 0, cx,
//     0, fy, cy,
//     0, 0, 1;
//     Kinv_ = K_.inverse();
// 
//     engine_ = std::make_shared<Renderer>(shape_[0], shape_[1]);
//     engine_->SetCamera(znear, zfar, fx, fy, cx, cy);
//     engine_->SetMesh(V_, F_);
//     Mat3 flip_co;   // object -> camera
//     flip_co << -1, 0, 0,
//         0, -1, 0,
//         0, 0, 1;
//     Mat3 flip_sc;   // camera -> spatial
//     flip_sc << 0, 0, 1,
//                -1, 0, 0,
//                0, -1, 0;
//     Mat3 flip_so{flip_sc * flip_co};   // object -> spatial
//     g_ = SE3(flip_so * g.so3().matrix(), flip_so * g.translation());
// 
//     BuildDistanceField();
// }


}   // namespace feh
