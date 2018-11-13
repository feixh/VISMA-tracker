#include <iostream>

#include "glog/logging.h"
#include "absl/strings/str_format.h"

#include "pix3d/dataloader.h"

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
  return disp;
}

int main(int argc, char **argv) {
    CHECK_EQ(argc, 2) << "requires root directory of pix3d as an argument!";
    Pix3dLoader loader(argv[1]);
    cv::namedWindow("image", CV_WINDOW_NORMAL);

    for (int i = 0; i < loader.size(); ++i) {
      auto packet = loader.GrabPacket(i); // index by path
      std::cout << "object pose=\n" << packet.go_.matrix3x4() << std::endl;
      std::cout << "object pose inverse=\n" << packet.go_.inv().matrix3x4() << std::endl;
      std::cout << "camera pose=\n" << packet.gc_.matrix3x4() << std::endl;
      std::cout << "camera pose inverse=" << packet.gc_.inv().matrix3x4() << std::endl;
      std::cout << "focal=" << packet.focal_length_ << std::endl;
      std::cout << "bbox=" << packet.bbox_.transpose() << std::endl;
      std::cout << "shape=" << packet.shape_.transpose() << std::endl;

      // // transfrom
      // packet.V_.col(0) *= -1;
      // packet.V_.col(1) *= -1;

      Vec3f xyz_max = packet.V_.colwise().maxCoeff();
      Vec3f xyz_min = packet.V_.colwise().minCoeff();

      // generate 9 control points: 8 corners of 3D Bounding Box + box centroid
      std::cout << "xyz_max=" << xyz_max.transpose() << std::endl;
      std::cout << "xyz_min=" << xyz_min.transpose() << std::endl;
      auto control_points = GenerateControlPoints(xyz_min, xyz_max);
      for (auto Xo : control_points) std::cout << Xo.transpose() << std::endl;


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
      for (auto xc : kps) std::cout << xc.transpose() << std::endl;
      cv::Mat disp = DrawBox(packet.img_, kps);

      cv::imshow("image", disp);
      char c = cv::waitKey();
      if (c == 'q') break;
    }
}

