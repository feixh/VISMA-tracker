// stl
#include <fstream>
#include <iostream>

// 3rd party
#include "glog/logging.h"
#include "sophus/se3.hpp"

// feh
#include "dataloaders.h"
#include "tracker.h"
#include "region_based_tracker.h"

using namespace feh;

int main(int argc, char **argv) {
    std::string config_file("../cfg/single_object_tracking.json");
    auto config = LoadJson(config_file);

    std::string dataset_root(config["dataset_root"].asString());

    if (argc == 1) {
        dataset_root += config["dataset"].asString();
    } else {
        dataset_root += std::string(argv[1]);
    }
    int wait_time(0);
    wait_time = config["wait_time"].asInt();

    VlslamDatasetLoader loader(dataset_root);

    tracker::Tracker tracker;
    tracker.Initialize(config["tracker_config"].asString());

    int start_index = config.get("start_index", 0).asInt();

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
            tracker.SetInitCameraToWorld(gwc);
        } else if (i < start_index) {
            continue;
        }


        if (config["method"] == "filtering") {
            // process
            if (!tracker.IsObjectPoseInitialized()) {
                for (const auto &bbox: bboxlist.bounding_boxes()) {
                    if (bbox.class_name() == "chair"
                        && bbox.scores(0) > 0.8) {
                        tracker.InitializeFromBoundingBox(bbox, gwc, Rg, imagepath);
                    }
                }
                if (tracker.IsObjectPoseInitialized()) {
                    tracker.Update(edgemap, bboxlist, gwc, Rg, img, imagepath);
                }
            }
            else {
                tracker.Update(edgemap, bboxlist, gwc, Rg, img, imagepath);
            }

            if (tracker.IsObjectPoseInitialized()) {
                cv::Mat display = tracker.GetFilterView();

                cv::imshow("tracker view", display);

                if (config["save"]["images"].asBool()) {
                    char ss[256];
                    sprintf(ss, "%s/%s_%06d.png",
                            config["save"]["root"].asString().c_str(),
                            config["save"]["tag"].asString().c_str(), i);
                    cv::imwrite(ss, display);
                }
                if (config["save"]["particles"].asBool()) {
                    char ss[256];
                    sprintf(ss, "%s/%s_%06d.txt",
                            config["save"]["root"].asString().c_str(),
                            config["save"]["tag"].asString().c_str(), i);
                    tracker.WriteOutParticles(ss);
                }

                char ckey = cv::waitKey(wait_time);
                if (ckey == 'q') {
                    break;
                }


            }
        } else if (config["method"] == "optimization") {

    }
}
}
