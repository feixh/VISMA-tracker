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
#include "gravity_aligned_tracker.h"

using namespace feh;

int main(int argc, char **argv) {
    std::string config_file("../cfg/DFTracker.json");
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
    tracker::NormalizeVertices(V);
    tracker::RotateVertices(V, -M_PI / 2);
    tracker::FlipVertices(V);

    std::string dataset_path(config["dataset_root"].asString() + config["dataset"].asString());

    int wait_time(0);
    wait_time = config["wait_time"].asInt();

    VlslamDatasetLoader loader(dataset_path);

    cv::namedWindow("tracker view", CV_WINDOW_NORMAL);
    cv::namedWindow("DF", CV_WINDOW_NORMAL);

    Sophus::SE3f camera_pose_t0;
    cv::Mat display;

    // initialization in camera frame
    Mat3 Rinit = Mat3::Identity();
    Vec3 Tinit = Vec3::Zero();
    Tinit = GetVectorFromDynamic<ftype, 3>(config, "Tinit");

    std::shared_ptr<GravityAlignedTracker> tracker{nullptr};



    Timer timer;
    for (int i = 0; i < loader.size(); ++i) {
        cv::Mat img, edgemap;
        vlslam_pb::BoundingBoxList bboxlist;
        Sophus::SE3f gwc;
        Sophus::SO3f Rg;

        std::string imagepath;
        bool success = loader.Grab(i, img, edgemap, bboxlist, gwc, Rg, imagepath);
        if (!success) break;

        std::cout << "gwc=\n" << gwc.matrix3x4() << std::endl;
        std::cout << "Rg=\n" << Rg.matrix() << std::endl;

        if (tracker == nullptr) {
            tracker = std::make_shared<GravityAlignedTracker>(
                img, edgemap,
                Vec2i{cam_cfg["rows"].asInt(), cam_cfg["cols"].asInt()},
                cam_cfg["fx"].asDouble(), cam_cfg["fy"].asDouble(),
                cam_cfg["cx"].asDouble(), cam_cfg["cy"].asDouble(),
                SE3{Rinit, Tinit},
                V, F);
            tracker->UpdateCameraPose(gwc);
            tracker->UpdateGravity(Rg);
        } else {
            tracker->UpdateImage(img, edgemap);
            tracker->UpdateCameraPose(gwc);
            tracker->UpdateGravity(Rg);
        }


        timer.Tick("tracking");
        float cost = tracker->Minimize(config["iterations"].asInt());
        timer.Tock("tracking");
        std::cout << timer;
        // std::cout << "cost=" << cost << std::endl;
        cv::imshow("tracker view", tracker->RenderEdgepixels());
        cv::imshow("DF", tracker->GetDistanceField());
        char ckey = cv::waitKey(wait_time);
        if (ckey == 'q') break;

//        // FIXME: CAN ONLY HANDLE CHAIR
//        for (int j = 0; j < bboxlist.bounding_boxes_size(); ) {
//            if (bboxlist.bounding_boxes(j).class_name() != "chair"
//                || bboxlist.bounding_boxes(j).scores(0) < 0.8) {
//                bboxlist.mutable_bounding_boxes()->DeleteSubrange(j, 1);
//            } else ++j;
//        }
    }

}
