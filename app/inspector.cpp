//
// Created by visionlab on 3/6/18.
//
// Frame Inspector
#include <io_utils.h>
#include "common/eigen_alias.h"
#include "dataset_loaders.h"
#include "tracker_utils.h"
#include "renderer.h"


// 3rd party
#include "opencv2/opencv.hpp"
#include "folly/dynamic.h"
#include "folly/FileUtil.h"
#include "folly/json.h"
#include "igl/readOBJ.h"

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

    // LOAD THE INPUT IMAGE
    int index = config["frame_inspector"]["index"].getInt();
    std::string basename = folly::sformat("./{}_{:06d}", config["dataset"].getString(), index);

    feh::VlslamDatasetLoader loader(dataset_path);
    Sophus::SE3f gwc;
    Sophus::SO3f Rg;
    cv::Mat img, edgemap;
    vlslam_pb::BoundingBoxList bboxlist;
    loader.Grab(index, img, edgemap, bboxlist, gwc, Rg);
    cv::imshow("input", img);

    // OVERLAY BOUNDING BOX PROPOSALS ON INPUT IMAGE
    cv::Mat input_with_proposals = img.clone();
    for (auto bbox : bboxlist.bounding_boxes()) {
        if (bbox.class_name() == "chair") {
            auto c = feh::tracker::kColorMap.at(bbox.class_name());
            cv::rectangle(input_with_proposals,
                          cv::Point((int)(bbox.top_left_x()), (int)(bbox.top_left_y())),
                          cv::Point((int)(bbox.bottom_right_x()), (int)(bbox.bottom_right_y())),
                          cv::Scalar(c[0], c[1], c[2]));
        }
    }
    cv::imshow("input image with proposals", input_with_proposals);
    cv::imwrite(basename + "_input.png", input_with_proposals);

    // INVERT EDGEMAP FOR PLEASING VISUALIZATION
    edgemap = 255 - edgemap;
    cv::imshow("edgemap", edgemap);
    cv::imwrite(basename + "_edgemap.png", edgemap);

    // Z-BUFFER COLOR-ENCODED BY INSTANCE LABEL
    folly::readFile((scene_dir + "/result.json").c_str(), contents);
    folly::dynamic result = folly::parseJson(folly::json::stripComments(contents)).at(index);
    // OVERWRITE SOME PARAMETERS
    std::vector<cv::Mat> depth_maps;
    feh::RendererPtr render_engine = std::make_shared<feh::Renderer>(img.rows, img.cols);
    {
        folly::readFile("../cfg/camera.json", contents);
        folly::dynamic camera = folly::parseJson(folly::json::stripComments(contents));
        // OVERWRITE SOME PARAMETERS
        float z_near = camera["z_near"].getDouble();
        float z_far = camera["z_far"].getDouble();
        float fx = camera["fx"].getDouble();
        float fy = camera["fy"].getDouble();
        float cx = camera["cx"].getDouble();
        float cy = camera["cy"].getDouble();
        render_engine->SetCamera(z_near, z_far, fx, fy, cx, cy);
//        render_engine->SetCamera(gwc.inverse().matrix());
        std::cout << "gwc=\n" << gwc.matrix() << "\n";
    }

    for (const auto &obj : result) {
        auto pose = feh::io::GetMatrixFromDynamic<float, 3, 4>(obj, "model_pose");
        std::cout << folly::format("id={}\nstatus={}\nshape={}\npose=\n",
                                   obj["id"].asInt(),
                                   obj["status"].asInt(),
                                   obj["model_name"].asString())
                  << pose << "\n";

        Sophus::SE3f gwm(pose.block<3,3>(0,0), pose.block<3,1>(0, 3));
        Sophus::SE3f gcm = gwc.inverse() * gwm;


        int instance_id = obj["id"].asInt();
        std::string model_name = obj["model_name"].asString();
        std::vector<float> v;
        std::vector<int> f;
        feh::io::LoadMeshFromObjFile(
            folly::sformat("{}/{}.obj", database_dir, model_name),
            v, f);
        render_engine->SetMesh(v, f);
        cv::Mat depth(render_engine->rows(), render_engine->cols(), CV_32FC1);
        render_engine->RenderDepth(gcm.matrix(), depth);
        cv::imshow(folly::sformat("depth#{}", instance_id), depth);
//        cv::Mat mask(render_engine->rows(), render_engine->cols(), CV_8UC1);
//        render_engine->RenderMask(gcm.matrix(), mask);
//        cv::imshow(folly::sformat("mask#{}", instance_id), mask);
    }

    // MEAN CONTOUR OVERLAID ON INPUT IMAGE

    cv::waitKey();

}

