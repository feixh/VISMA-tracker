// Utilities for communication, e.g., parsing/loading of protobuf messages
// creation & binding of zmq sockets
#pragma once
#include <string>
#include <vector>
#include "vlslam.pb.h"
#include "opencv2/core/core.hpp"

namespace feh {

/// \brief: Load edgemap from protobuf file.
bool LoadEdgeMap(const std::string &filename, cv::Mat &edge);

/// \brief: Draw bounding boxes on the input image and return an image.
cv::Mat DrawBoxList(const cv::Mat &image, const vlslam_pb::NewBox &box);
/// \brief: Draw a list of bounding boxes on the input image.
cv::Mat DrawBoxList(const cv::Mat &image, const vlslam_pb::NewBoxList &boxlist);

} // namespace feh

