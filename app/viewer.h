//
// Created by visionlab on 3/18/18.
//
#pragma once


#include "renderer.h"
#include "vlslam.pb.h"

// 3rd party
#include "opencv2/opencv.hpp"
#include "folly/dynamic.h"
#include "sophus/se3.hpp"

void DrawOneFrame(const cv::Mat &img,
                  const cv::Mat &edgemap,
                  const vlslam_pb::BoundingBoxList &bboxlist,
                  const Sophus::SE3f &gwc,
                  const Sophus::SO3f &Rg,
                  feh::RendererPtr render_engine,
                  const folly::dynamic &config,
                  const folly::dynamic &result, // result at this time stamp
                  cv::Mat *ptr_input_with_proposals=nullptr,
                  cv::Mat *ptr_edgemap=nullptr,
                  cv::Mat *ptr_input_with_contour=nullptr);


