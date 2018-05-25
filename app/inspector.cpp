//
// Created by visionlab on 3/6/18.
//
// Frame Inspector
#include <io_utils.h>
#include "dataset_loaders.h"
#include "tracker_utils.h"
#include "renderer.h"
#include "viewer.h"

// 3rd party
#include "folly/FileUtil.h"
#include "folly/json.h"


int main(int argc, char **argv) {
    CHECK_EQ(argc, 3);

    folly::fbstring contents;
    folly::readFile("../cfg/tool.json", contents);
    folly::dynamic config = folly::parseJson(folly::json::stripComments(contents));
    // OVERWRITE SOME PARAMETERS
    config["dataset"] = std::string(argv[1]);
    config["frame_inspector"]["index"] = folly::to<int>(argv[2]);

    // EXTRACT PATHS
    std::string dataset_path = folly::sformat("{}{}", config["experiment_root"].getString(), config["dataset"].getString());
    std::string database_dir = config["CAD_database_root"].getString();
    std::string scene_dir = config["dataroot"].getString() + "/" + config["dataset"].getString();

    // setup loader
    std::shared_ptr<feh::VlslamDatasetLoader> loader;
    if (config["datatype"].getString() == "VLSLAM") {
        loader = std::make_shared<feh::VlslamDatasetLoader>(dataset_path);
        folly::readFile("../cfg/camera.json", contents);
    } else if (config["datatype"].getString() == "SceneNN") {
        loader = std::make_shared<feh::SceneNNDatasetLoader>(dataset_path);
        folly::readFile("../cfg/camera_scenenn.json", contents);
    }
    // load camera
    folly::dynamic camera;
    if (config["datatype"].getString() == "VLSLAM") {
        folly::readFile("../cfg/camera.json", contents);
        camera = folly::parseJson(folly::json::stripComments(contents));
    } else if (config["datatype"].getString() == "SceneNN") {
        folly::readFile("../cfg/camera_scenenn.json", contents);
        camera = folly::parseJson(folly::json::stripComments(contents));
    }
    // setup renders
    feh::RendererPtr render_engine = std::make_shared<feh::Renderer>(camera["rows"].getInt(), camera["cols"].getInt());
    {

        // OVERWRITE SOME PARAMETERS
        float z_near = camera["z_near"].getDouble();
        float z_far = camera["z_far"].getDouble();
        float fx = camera["fx"].getDouble();
        float fy = camera["fy"].getDouble();
        float cx = camera["cx"].getDouble();
        float cy = camera["cy"].getDouble();
        render_engine->SetCamera(z_near, z_far, fx, fy, cx, cy);
    }
    // holders for variables
    Sophus::SE3f gwc;
    Sophus::SO3f Rg;
    cv::Mat img, edgemap;
    vlslam_pb::BoundingBoxList bboxlist;

    // LOAD THE INPUT IMAGE
    int index = config["frame_inspector"]["index"].getInt();

//    if (index >= 0)
    {
        folly::readFile((scene_dir + "/result.json").c_str(), contents);
        // FIXME: RESULT FILE SHOULD KEEP TRACK OF TIMESTAMP
        folly::dynamic result = folly::parseJson(folly::json::stripComments(contents)).at(index); //-130);

        std::string basename = folly::sformat("./{}_{:06d}", config["dataset"].getString(), index);
        loader->Grab(index, img, edgemap, bboxlist, gwc, Rg);

        cv::Mat input_with_proposals, inverse_edgemap, input_with_contour;
        DrawOneFrame(img, edgemap, bboxlist, gwc, Rg, render_engine, config, result,
                     &input_with_proposals,
                     &inverse_edgemap,
                     &input_with_contour);
        cv::imshow("input image with proposals", input_with_proposals);
        cv::imwrite(basename + "_input.png", input_with_proposals);

        cv::imshow("edgemap", inverse_edgemap);
        cv::imwrite(basename + "_edgemap.png", inverse_edgemap);

        cv::imshow("segmask", input_with_contour);
        cv::imwrite(basename + "_mask.png", input_with_contour);

        cv::waitKey();

    }
}

