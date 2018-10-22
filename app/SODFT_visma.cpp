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
#include "pix3d/diff_tracker.h"

using namespace feh;

int main(int argc, char **argv) {
    std::string config_file("../cfg/single_object_tracking.json");
    if (argc > 1) {
        config_file = argv[1];
    }

    std::string content;
    folly::readFile(config_file.c_str(), content);
    auto config = folly::parseJson(folly::json::stripComments(content));

    folly::readFile(config["camera_config"].asString().c_str(), content);
    auto cam_cfg = folly::parseJson(folly::json::stripComments(content));

    MatXf V;
    MatXi F;
    LoadMesh(config["CAD_model"].asString(), V, F);

    std::string dataset_root(config["dataset_root"].asString());

    int wait_time(0);
    wait_time = config["wait_time"].asInt();

    VlslamDatasetLoader loader(dataset_root);

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

            
        Mat3 Rinit;
        Vec3 Tinit;
        DiffTracker tracker(img, edgemap,
                Vec2i{cam_cfg["rows"].asInt(), cam_cfg["cols"].asInt()},
                cam_cfg["fx"].asDouble(), cam_cfg["fy"].asDouble(), 
                cam_cfg["cx"].asDouble(), cam_cfg["cy"].asDouble(),
                Rinit, Tinit,
                V, F);

        // FIXME: CAN ONLY HANDLE CHAIR
        for (int j = 0; j < bboxlist.bounding_boxes_size(); ) {
            if (bboxlist.bounding_boxes(j).class_name() != "chair"
                || bboxlist.bounding_boxes(j).scores(0) < 0.8) {
                bboxlist.mutable_bounding_boxes()->DeleteSubrange(j, 1);
            } else ++j;
        }

        if (i == 0) {
            // tracker.SetInitCameraToWorld(gwc);
        } else if (i < start_index) {
            continue;
        }
}

}
