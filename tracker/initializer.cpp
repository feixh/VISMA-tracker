#include <iostream>

#include "initializer.h"
#include "opencv2/calib3d/calib3d.hpp"

namespace feh {

SE3 Initialize(const std::vector<Vec3> &X, 
               const std::vector<Vec2> &x, 
               const std::vector<bool> &v,
               const Mat3 &K) {
  std::vector<Vec3f> XX;
  std::vector<Vec2f> xx;
  for (int i = 0; i < v.size(); ++i) 
    if (v[i]) {
      XX.push_back(X[i]);
      xx.push_back(x[i]);
    }
  return Initialize(XX, xx, K);
}

SE3 Initialize(const std::vector<Vec3> &X, 
               const std::vector<Vec2> &x, 
               const Mat3 &K) {
  std::vector<cv::Point3f> cvX;
  for (auto pt : X) {
    cvX.push_back(cv::Point3f(pt(0), pt(1), pt(2)));
    // std::cout << cvX.back() << std::endl;
  }

  std::vector<cv::Point2f> cvx;
  for (auto pt : x) {
    cvx.push_back(cv::Point2f(pt(0), pt(1)));
    // std::cout << cvx.back() << std::endl;
  }
  assert(cvX.size() == cvx.size());

  // opencv calibration matrix
  cv::Mat cvK(3, 3, CV_32FC1);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      cvK.at<float>(i, j) = K(i, j);
    }

  // cv::Mat rvec(3, 1, CV_32FC1), tvec(3, 1, CV_32FC1);
  cv::Mat rvec, tvec;
  cv::solvePnP(cvX, cvx, cvK, cv::Mat{}, rvec, tvec);

  // std::cout << "tvec=" << tvec << std::endl;
  // std::cout << "rvec=" << rvec << std::endl;

  Vec3 t, w;
  for (int i = 0; i < 3; ++i) {
    // NOTE: double precision ... not know until trials
    t(i) = tvec.at<double>(i, 0);
    w(i) = rvec.at<double>(i, 0);
  }

  auto R = rodrigues(w);
  return {R, t};
}

}
