// Frame Inspector
#include <io_utils.h>
#include "dataset_loaders.h"
#include "tracker_utils.h"
#include "renderer.h"
#include "viewer.h"

// 3rd party
#include "opencv2/opencv.hpp"
#include "folly/FileUtil.h"
#include "folly/json.h"
#include "tbb/parallel_for.h"

// ros
#include "ros/ros.h"
#include "cv_bridge/cv_bridge.h"

int main(int argc, char **argv) {
    // setup rosnode
    ros::init(argc, argv, "rosviewer");
    ros::NodeHandle nh("~");
    ros::Publisher proposal_pub = nh.advertise<sensor_msgs::Image>("vismap/proposal", 1);
    ros::Publisher mask_pub = nh.advertise<sensor_msgs::Image>("vismap/mask", 1);
    std::string proj_root;
    nh.getParam("proj_root", proj_root);
//    std::cout << "==========\n===============\n=================\n project root=" << proj_root << "\n======================\n====================\n==================\n";

    folly::fbstring contents;
    folly::readFile((proj_root + "/cfg/tool.json").c_str(), contents);
    folly::dynamic config = folly::parseJson(folly::json::stripComments(contents));
    // OVERWRITE SOME PARAMETERS
//    config["dataset"] = std::string(argv[1]);
//    config["frame_inspector"]["index"] = folly::to<int>(argv[2]);
//    std::cout << "==========\n===============\n=================\n tool.json loaded \n======================\n====================\n==================\n";

    // EXTRACT PATHS
    std::string
        dataset_path = folly::sformat("{}{}", config["experiment_root"].getString(), config["dataset"].getString());
    std::string database_dir = config["CAD_database_root"].getString();
    std::string scene_dir = config["dataroot"].getString() + "/" + config["dataset"].getString();

    // setup loader
    std::shared_ptr <feh::VlslamDatasetLoader> loader;
    if (config["datatype"].getString() == "VLSLAM") {
        loader = std::make_shared<feh::VlslamDatasetLoader>(dataset_path);
    } else if (config["datatype"].getString() == "SceneNN") {
        loader = std::make_shared<feh::SceneNNDatasetLoader>(dataset_path);
    }
//    std::cout << "==========\n===============\n=================\n dataset loader set \n======================\n====================\n==================\n";

    // load camera
    folly::dynamic camera;
    if (config["datatype"].getString() == "VLSLAM") {
        folly::readFile((proj_root + "/cfg/camera.json").c_str(), contents);
        camera = folly::parseJson(folly::json::stripComments(contents));
    } else if (config["datatype"].getString() == "SceneNN") {
        folly::readFile((proj_root + "/cfg/camera_scenenn.json").c_str(), contents);
        camera = folly::parseJson(folly::json::stripComments(contents));
    }
//    std::cout << "==========\n===============\n=================\n camera loaded \n======================\n====================\n==================\n";
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
    folly::readFile((scene_dir + "/result.json").c_str(), contents);
    folly::dynamic results = folly::parseJson(folly::json::stripComments(contents));
    for (int index = 0; index < loader->size(); ++index) {
        auto result = results.at(index);
        std::string basename = folly::sformat("./{}_{:06d}", config["dataset"].getString(), index);
        loader->Grab(index, img, edgemap, bboxlist, gwc, Rg);

        cv::Mat input_with_proposals, inverse_edgemap, input_with_contour;
        DrawOneFrame(img, edgemap, bboxlist, gwc, Rg, render_engine, config, result,
                     &input_with_proposals,
                     &inverse_edgemap,
                     &input_with_contour);
//            cv::imshow("input image with proposals", input_with_proposals);
//            cv::imwrite(basename + "_input.png", input_with_proposals);

//            cv::imshow("edgemap", inverse_edgemap);
//            cv::imwrite(basename + "_edgemap.png", inverse_edgemap);

//            cv::imshow("segmask", input_with_contour);
//            cv::imwrite(basename + "_mask.png", input_with_contour);
//            char ckey = cv::waitKey(24);
//            if (ckey == 'q') {
//                break;
//            }

        cv_bridge::CvImage proposal_img;
//            proposal_img.header.stamp = ;
        proposal_img.header.frame_id = "proposal";
        proposal_img.encoding = "bgr8";
        proposal_img.image = input_with_proposals;
        sensor_msgs::Image proposal_img_msg;
        proposal_img.toImageMsg(proposal_img_msg);
        proposal_pub.publish(proposal_img_msg);
//        std::cout << "input image with proposals published\n";

        cv_bridge::CvImage mask_img;
//            mask_img.header.stamp = ;
        mask_img.header.frame_id = "mask";
        mask_img.encoding = "bgr8";
        mask_img.image = input_with_contour;
        sensor_msgs::Image mask_img_msg;
        mask_img.toImageMsg(mask_img_msg);
        mask_pub.publish(mask_img_msg);
//        std::cout << "input image with masks published\n";

        ros::spinOnce();
        sleep(0.03);
    }
}

