// Frame Inspector
#include <io_utils.h>
#include "dataset_loaders.h"
#include "tracker_utils.h"
#include "renderer.h"
#include "viewer.h"

// 3rd party
#include "folly/FileUtil.h"
#include "folly/json.h"

// ros
#include "ros/ros.h"
#include "cv_bridge/cv_bridge.h"
#include "visualization_msgs/Marker.h"

int main(int argc, char **argv) {
    // setup rosnode
    ros::init(argc, argv, "rosviewer");
    ros::NodeHandle nh("~");
    ros::Publisher proposal_pub = nh.advertise<sensor_msgs::Image>("visma/proposal", 1);
    ros::Publisher mask_pub = nh.advertise<sensor_msgs::Image>("visma/mask", 1);
    ros::Publisher obj_pub = nh.advertise<visualization_msgs::Marker>("visma/object", 10);
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


        // draw objects
        {

                auto cm = feh::GenerateRandomColorMap<8>();
                cm[0] = {255, 255, 255};    // white background

                for (const auto &obj : result) {
                    auto pose = feh::io::GetMatrixFromDynamic<float, 3, 4>(obj, "model_pose");
//            std::cout << folly::format("id={}\nstatus={}\nshape={}\npose=\n",
//                                       obj["id"].asInt(),
//                                       obj["status"].asInt(),
//                                       obj["model_name"].asString())
//                      << pose << "\n";

                    Sophus::SE3f gwm(pose.block<3,3>(0,0), pose.block<3,1>(0, 3));
                    Sophus::SE3f gcm = gwc.inverse() * gwm;

                    // translation and quaternion of gwm
                    auto ts = gwm.translation();
                    auto qs = gwm.so3().unit_quaternion();

                    int instance_id = obj["id"].asInt();
                    std::string model_name = obj["model_name"].asString();
                    auto color = cm[instance_id+1];
//                    std::vector<float> v;
//                    std::vector<int> f;
//                    feh::io::LoadMeshFromObjFile(
//                        folly::sformat("{}/{}.obj", database_dir, model_name),
//                        v, f);
//                    render_engine->SetMesh(v, f);
//                    cv::Mat depth(size, CV_32FC1);
//                    depth.setTo(0);
//                    render_engine->RenderDepth(gcm.matrix(), depth);
//                    feh::tracker::PrettyDepth(depth);
//                    cv::Mat contour(size, CV_8UC1);
//                    render_engine->RenderEdge(gcm.matrix(), contour);
                    visualization_msgs::Marker marker;
                    marker.header.frame_id = "/map";
                    marker.ns = "ns";
                    marker.id = instance_id;
                    marker.type = marker.MESH_RESOURCE;
                    marker.action = marker.MODIFY;
                    marker.pose.position.x = ts(0);
                    marker.pose.position.y = ts(1);
                    marker.pose.position.z = ts(2);
                    marker.pose.orientation.x = qs.x();
                    marker.pose.orientation.y = qs.y();
                    marker.pose.orientation.z = qs.z();
                    marker.pose.orientation.w = qs.w();
                    marker.scale.x = 1;
                    marker.scale.y = 1;
                    marker.scale.z = 1;
                    marker.color.a = 1;
                    marker.color.r = color[0] / 255.0;
                    marker.color.g = color[1] / 255.0;
                    marker.color.b = color[2] / 255.0;
                    std::string uri = "file://" + database_dir + "/" + model_name + ".obj";
                    marker.mesh_resource = uri;
//                    std::cout << "uri=" << uri << "\n";

                    obj_pub.publish(marker);

                }
        }


        ros::spinOnce();
        sleep(0.03);
    }
}

