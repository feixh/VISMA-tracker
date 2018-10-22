// stl
#include <fstream>
#include <iostream>

// 3rd party
#include "glog/logging.h"
#include "sophus/se3.hpp"
#include "folly/json.h"
#include "folly/FileUtil.h"

// feh
#include "tracker.h"
#include "tracker_utils.h"
#include "dataloaders.h"
#include "region_based_tracker.h"

int main(int argc, char **argv) {
    std::string config_file("../cfg/single_object_tracking.json");

    std::ifstream in_config(config_file);
    if (!in_config.is_open()) {
        LOG(FATAL) << "FATAL::failed to open config file @ " << config_file;
    }
//    Json::Value config;
//    in_config >> config;
//    in_config.close();
    std::string content;
    folly::readFile(config_file.c_str(), content);
    folly::dynamic config = folly::parseJson(folly::json::stripComments(content));


    std::string dataset_root(config["dataset_root"].asString());

    if (argc == 1) {
        dataset_root += config["dataset"].asString();
    } else {
        dataset_root += std::string(argv[1]);
    }
    int wait_time(0);
    wait_time = config["wait_time"].asInt();

    feh::VlslamDatasetLoader loader(dataset_root);

//    tracker.Initialize(config["tracker_config"].asString());

    int start_index = config.getDefault("start_index", 0).asInt();

    cv::namedWindow("tracker view", CV_WINDOW_NORMAL);
    Sophus::SE3f camera_pose_t0;
    cv::Mat display;
    for (int i = start_index; i < loader.size(); ++i) {
        cv::Mat img, edgemap;
        vlslam_pb::BoundingBoxList bboxlist;
        Sophus::SE3f gwc;
        Sophus::SO3f Rg;

        std::string imagepath;
        loader.Grab(i, img, edgemap, bboxlist, gwc, Rg, imagepath);

        // FIXME: CAN ONLY HANDLE CHAIR
        for (int j = 0; j < bboxlist.bounding_boxes_size(); ) {
            if (bboxlist.bounding_boxes(j).class_name() != "chair"
                || bboxlist.bounding_boxes(j).scores(0) < 0.8) {
                bboxlist.mutable_bounding_boxes()->DeleteSubrange(j, 1);
            } else ++j;
        }

//        cv::Mat gray;
//        cv::cvtColor(img, gray, CV_RGB2GRAY);
//        cv::blur(gray, edgemap, cv::Size(3, 3));
//        cv::Canny(edgemap, edgemap, 100, 200);
//        cv::Mat dst(cv::Scalar::all(0));
//        gray.copyTo(dst, edgemap);
//        edgemap = dst;

        if (i == 0) {
            // tracker.SetInitCameraToWorld(gwc);
        } else if (i < start_index) {
            continue;
        }
}

}
