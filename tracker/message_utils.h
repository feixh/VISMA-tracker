// Utilities for communication, e.g., parsing/loading of protobuf messages
// creation & binding of zmq sockets
#pragma once
#include <string>
#include <vector>
#include "opencv2/core/core.hpp"

#include "vlslam.pb.h"
#include "alias.h"

namespace feh {

/// \brief: Load edgemap from protobuf file.
bool LoadEdgeMap(const std::string &filename, cv::Mat &edge);

/// \brief: Draw bounding boxes on the input image and return an image.
cv::Mat DrawBoxList(const cv::Mat &image, const vlslam_pb::NewBox &box);
/// \brief: Draw a list of bounding boxes on the input image.
cv::Mat DrawBoxList(const cv::Mat &image, const vlslam_pb::NewBoxList &boxlist);

std::vector<Vec2> KeypointsFromBox(const vlslam_pb::NewBox &box, int rows=500, int cols=960);

} // namespace feh

