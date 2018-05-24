//
// Created by visionlab on 5/23/18.
//


// 3rd party
#include "sophus/se3.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"

// own
#include "vlslam.pb.h"
#include "common/eigen_alias.h"
#include "dataset_loaders.h"

using namespace feh;

int main() {
    KittiDatasetLoader loader("/local/feixh/experiments/kitti00/");
    for (int i = 0; i < loader.size(); ++i) {
        std::cout << i << "/" << loader.size() << std::endl;
        cv::Mat image, edgemap;
        vlslam_pb::BoundingBoxList bboxlist;
        Sophus::SE3f gwc;
        Sophus::SO3f Rg;
        loader.Grab(i, image, edgemap, bboxlist, gwc, Rg);
        cv::imshow("image", image);
        cv::imshow("edge", edgemap);
        cv::waitKey(24);
    }
}
