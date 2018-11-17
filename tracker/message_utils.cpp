#include "message_utils.h"
#include "utils.h"

namespace feh {

bool LoadEdgeMap(const std::string &filename, cv::Mat &edge) {
    vlslam_pb::EdgeMap edgemap;
    try {
        std::ifstream in_file(filename);
        edgemap.ParseFromIstream(&in_file);
        in_file.close();
        edge = cv::Mat(edgemap.rows(), edgemap.cols(),
                       CV_32FC1,
                       edgemap.mutable_data()->mutable_data());
        edge.convertTo(edge, CV_8UC1, 255.0f);
        return true;
    } catch (const std::exception &e) {
        return false;
    }
}



cv::Mat DrawBoxList(const cv::Mat &image, const vlslam_pb::NewBox &box) {
  static std::vector<std::pair<int, int>> edges{{0, 1}, {0, 2}, {0, 4}, {1, 3}, 
                                      {1, 5}, {2, 6}, {2, 3}, {4, 6},
                                      {4, 5}, {3, 7}, {6, 7}, {5, 7}};
  cv::Mat disp(image.clone());
  int rows = image.rows;
  int cols = image.cols;
  cv::rectangle(disp, 
      cv::Point(box.top_left_x()*cols, box.top_left_y()*rows), 
      cv::Point(box.bottom_right_x()*cols, box.bottom_right_y()*rows), 
      cv::Scalar(255, 0, 0), 4);
  if (box.keypoints_size()) {
    for (int i = 0; i < edges.size(); ++i) {
      int idx1 = edges[i].first;
      int idx2 = edges[i].second;
      cv::Point pt1(cols*box.keypoints(idx1*2), rows*box.keypoints(idx1*2+1));
      cv::Point pt2(cols*box.keypoints(idx2*2), rows*box.keypoints(idx2*2+1));
      cv::line(disp, pt1, pt2, cv::Scalar(255, 0, 0), 2);
    }
  }
  return disp;
}


cv::Mat DrawBoxList(const cv::Mat &image, const vlslam_pb::NewBoxList &boxlist) {
  cv::Mat disp(image.clone());
  int rows = image.rows;
  int cols = image.cols;
  for (auto box : boxlist.boxes()) {
    // Not very efficient, but clear ...
    disp = DrawBoxList(disp, box);
  }
  return disp;
}

}
