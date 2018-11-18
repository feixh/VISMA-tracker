#include "initializer.h"
#include "opencv2/calib3d/calib3d.hpp"

namespace feh {

SE3 Initialize(const std::vector<Vec3> &X, 
               const std::vector<Vec2> &x, 
               const Mat3 &K) {
  std::vector<cv::Point3f> cvX;
  for (auto pt : X) {
    cvX.push_back(cv::Point3f(pt(0), pt(1), pt(2)));
  }

  std::vector<cv::Point2f> cvx;
  for (auto pt : x) {
    cvx.push_back(cv::Point2f(pt(0), pt(1)));
  }
  assert(cvX.size() == cvx.size());

  // opencv calibration matrix
  cv::Mat cvK(3, 3, CV_32FC1);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      cvK.at<float>(i, j) = K(i, j);
    }

  cv::Mat rvec, tvec;
  cv::solvePnP(cvX, cvx, cvK, cv::Mat{}, rvec, tvec);

  Vec3 t, w;
  for (int i = 0; i < 3; ++i) {
    t(i) = tvec.at<float>(i);
    w(i) = rvec.at<float>(i);
  }

  auto R = rodrigues(w);
  return {R, t};
}

}
