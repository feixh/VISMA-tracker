#define STRIP_FLAG_HELP 1
#include <iostream>

#include "glog/logging.h"
#include "gflags/gflags.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

#include "pix3dloader.h"
#include "tracker_utils.h"
#include "initializer.h"

DEFINE_string(pix3d_root, "", "Root directory of Pix3d dataset.");
DEFINE_int32(wait_time, 5, "Wait time for the opencv window.");
DEFINE_bool(show_control_points, true, "If true, visualize the control points.");
DEFINE_bool(save_control_points, false, "If true, save the control points.");

using namespace feh; 

auto ColorMap = GenerateRandomColorMap();

cv::Mat DrawBox(const cv::Mat &img, const std::vector<Vec2> &xc) {
  cv::Mat disp = img.clone();
  for (int i = 0; i < xc.size(); ++i) {
    auto x = xc[i];
    cv::circle(disp, cv::Point(int(x(0)), int(x(1))), 5, 
        cv::Scalar(ColorMap[i][0], ColorMap[i][1], ColorMap[i][2]), -1);
  }
  std::vector<std::pair<int, int>> edges = {{0, 1}, {0, 2}, {0, 4}, {1, 3}, 
                                          {1, 5}, {2, 6}, {2, 3}, {4, 6}, 
                                          {4, 5}, {3, 7}, {6, 7}, {5, 7}};
  for (auto e : edges) {
    cv::Point p1(xc[e.first](0), xc[e.first](1));
    cv::Point p2(xc[e.second](0), xc[e.second](1));
    cv::line(disp, p1, p2, cv::Scalar(255, 0, 0), 2);
  }
  return disp;
}

int main(int argc, char **argv) {
    gflags::SetUsageMessage("generate virtual control points for pix3d dataset");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    Pix3dLoader loader(FLAGS_pix3d_root);
    cv::namedWindow("image", CV_WINDOW_NORMAL);

    for (int i = 0; i < loader.size(); ++i) {
      std::cout << absl::StrFormat("%05d/%05d", i, loader.size()) << std::endl;
      auto packet = loader.GrabPacket(i); // index by path
      // std::cout << "object pose=\n" << packet.go_.matrix3x4() << std::endl;
      // std::cout << "object pose inverse=\n" << packet.go_.inv().matrix3x4() << std::endl;
      // std::cout << "camera pose=\n" << packet.gc_.matrix3x4() << std::endl;
      // std::cout << "camera pose inverse=" << packet.gc_.inv().matrix3x4() << std::endl;
      // std::cout << "focal=" << packet.focal_length_ << std::endl;
      // std::cout << "bbox=" << packet.bbox_.transpose() << std::endl;
      // std::cout << "shape=" << packet.shape_.transpose() << std::endl;
      
      // generate 9 control points: 8 corners of 3D Bounding Box + box centroid
      auto control_points = tracker::GenerateControlPoints(packet.V_);
      // for (auto Xo : control_points) std::cout << Xo.transpose() << std::endl;

      // project
      std::vector<Vec2f> kps; // keypoints on image plane, in pixel coordinates
      Mat3 Flip;
      Flip << -1, 0, 0,
           0, -1, 0,
           0, 0, 1;

      Mat3 Rgt = Flip * packet.go_.so3().matrix();
      Vec3 Tgt = Flip * packet.go_.translation();
      SE3 g_gt{Rgt, Tgt};

      for (auto Xo : control_points) {
        auto Xc = g_gt * Xo;
        Xc /= Xc(2);
        Xc = packet.K_ * Xc;
        auto xc = Xc.head<2>();
        kps.push_back(xc);
      }
      // for (auto xc : kps) std::cout << xc.transpose() << std::endl;

      if (FLAGS_save_control_points) {
        std::string path = packet.record_["img"].asString();
        path.erase(path.find_last_of('.'));
        path = absl::StrCat(argv[1], "/", path, "_virtual_control_points.txt");
        // std::cout << path << std::endl;
        std::ofstream ofs(path, std::ios::out);
        assert(ofs.is_open());
        for (auto xc : kps) ofs << xc.transpose() << std::endl;
        ofs.close();
      }

      if (FLAGS_show_control_points) {
        cv::Mat disp = DrawBox(packet.img_, kps);
        cv::imshow("image", disp);
        char c = cv::waitKey(FLAGS_wait_time);
        if (c == 'q') break;
      }

      // test the pose initializer given 3D control points and
      // 2D projections
      auto est = Initialize(control_points, kps, packet.K_);
      std::cout << "==========\n";
      std::cout << "est=\n" << est.matrix3x4() << std::endl;
      std::cout << "gt=\n" << g_gt.matrix3x4() << std::endl;
    }
}

