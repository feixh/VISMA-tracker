#include <fstream>
#include <iostream>

#include "glog/logging.h"
#include "sophus/se3.hpp"
#include "folly/json.h"
#include "folly/FileUtil.h"

// feh
#include "io_utils.h"
#include "tracker.h"
#include "tracker_utils.h"
#include "scene.h"
#include "dataset_loaders.h"

int main(int argc, char **argv) {
    std::string config_file("../cfg/multiple_object_tracking.json");

    std::string content;
    folly::readFile(config_file.c_str(), content);
    folly::dynamic config = folly::parseJson(folly::json::stripComments(content));

    std::string dataset_root(config["dataset_root"].asString() + "/");

    std::string dataset;
    if (argc == 1) {
        dataset = config["dataset"].asString();
    } else {
        dataset = std::string(argv[1]);
    }
    dataset_root += dataset;
    int wait_time(0);
    wait_time = config["wait_time"].asInt();


    std::shared_ptr<feh::VlslamDatasetLoader> loader;
    if (config["datatype"].getString() == "VLSLAM") {
        std::cout << dataset_root << "\n";
        loader = std::make_shared<feh::VlslamDatasetLoader>(dataset_root);
    } else if (config["datatype"].getString() == "ICL") {
        std::cout << dataset_root << "\n";
        loader = std::make_shared<feh::ICLDatasetLoader>(dataset_root);
    }

    feh::tracker::Scene scene;
    scene.Initialize(config["scene_config"].asString());

    // create windows
    int start_index(config["start_index"].asInt());
    char *dump_dir = nullptr;

    if (config["save"].asBool()) {
        char temp_template[256];
        sprintf(temp_template, "%s_XXXXXX", dataset.c_str());
        dump_dir = mkdtemp(temp_template);
    }


    for (int i = 0; i < loader->size(); ++i) {
        std::cout << "outer loop " <<  i << "/" << loader->size() << "\n";
        cv::Mat img, edgemap;
        vlslam_pb::BoundingBoxList bboxlist;
        Sophus::SE3f gwc;
        Sophus::SO3f Rg;

        std::string imagepath;
        loader->Grab(i, img, edgemap, bboxlist, gwc, Rg, imagepath);
        std::cout << imagepath << "\n";
        if (i == 0) {
            // global reference frame
            scene.SetInitCameraToWorld(gwc);
        } else if (i < start_index) {
            continue;
        }

        scene.Update(edgemap, bboxlist, gwc, Rg, img, imagepath);
//        scene.Update(edgemap, pruned_bboxlist, gwc, Rg, img);
        const auto &display = scene.Get2DView();
        const auto &zbuffer = scene.GetZBuffer();
        const auto &segmask = scene.GetSegMask();

        cv::imshow("tracker view", display);
//        cv::imshow("depth buffer", zbuffer);
//        cv::imshow("segmentation mask", segmask);

        if (dump_dir != nullptr) {
            cv::imwrite(folly::sformat("{}/{:06d}.png", dump_dir, i),
                        display);
        }

        char ckey = cv::waitKey(wait_time);
        if (ckey == 'q') {
            break;
        } else if (ckey == 'p') {
            // pause
            wait_time = 0;
        } else if (ckey == 'r') {
            wait_time = config["wait_time"].asInt();
        } else if (ckey == 's') {
            if (dump_dir != nullptr) {
                scene.WriteLogToFile(folly::sformat("{}/result_{:04d}.json", dump_dir, i));
            }
        }
    }
    if (dump_dir != nullptr) {
        scene.WriteLogToFile(folly::sformat("{}/result.json", dump_dir));
    }
}
