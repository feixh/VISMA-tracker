#define STRIP_FLAG_HELP 1
#include <iostream>

#include "glog/logging.h"
#include "gflags/gflags.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"

#include "pix3dloader.h"

DEFINE_string(pix3d_root, "", "Root directory of Pix3d dataset.");
DEFINE_int32(wait_time, 5, "Wait time for the opencv window.");
DEFINE_bool(show_control_points, true, "If true, visualize the control points.");
DEFINE_bool(save_control_points, false, "If true, save the control points.");

using namespace feh; 

auto ColorMap = GenerateRandomColorMap();

std::vector<Vec3f> GenerateControlPoints(const Vec3f &xyz_min, const Vec3f &xyz_max) {
  std::vector<Vec3f> xyz{xyz_min, xyz_max};
  std::vector<Vec3f> out;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 2; ++k)
        out.push_back({xyz[i](0), xyz[j](1), xyz[k](2)});
  out.push_back(0.5 * (xyz_min + xyz_max));
  return out;
};

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

      // // transfrom
      // packet.V_.col(0) *= -1;
      // packet.V_.col(1) *= -1;

      Vec3f xyz_max = packet.V_.colwise().maxCoeff();
      Vec3f xyz_min = packet.V_.colwise().minCoeff();

      // generate 9 control points: 8 corners of 3D Bounding Box + box centroid
      // std::cout << "xyz_max=" << xyz_max.transpose() << std::endl;
      // std::cout << "xyz_min=" << xyz_min.transpose() << std::endl;
      auto control_points = GenerateControlPoints(xyz_min, xyz_max);
      // for (auto Xo : control_points) std::cout << Xo.transpose() << std::endl;

      // project
      std::vector<Vec2f> kps; // keypoints on image plane, in pixel coordinates
      for (auto Xo : control_points) {
        auto Xc = packet.go_ * Xo;
        Xc(0) *= -1;
        Xc(1) *= -1;
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
    }
}

