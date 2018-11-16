// stl
#include <fstream>
#include <iostream>

// 3rd party
#include "glog/logging.h"
#include "json/json.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "zmqpp/zmqpp.hpp"

// feh
#include "tracker.h"
#include "tracker_utils.h"
#include "dataloaders.h"
#include "gravity_aligned_tracker.h"
#include "vlslam.pb.h"

using namespace feh;

namespace feh {

cv::Mat DrawBoxList(const cv::Mat &image, const vlslam_pb::NewBox &box) {
  static std::vector<std::pair<int, int>> edges{{0, 1}, {0, 2}, {0, 4}, {1, 3}, 
                                      {1, 5}, {2, 6}, {2, 3}, {4, 6},
                                      {4, 5}, {3, 7}, {6, 7}, {5, 7}};
  cv::Mat disp(image.clone());
  int rows = image.rows;
  int cols = image.cols;
  cv::rectangle(disp, 
      cv::Point(box.top_left_x()*cols, box.top_left_y()*rows), 
      cv::Point(box.bottom_right_x()*cols, box.bottom_right_y()*rows), 
      cv::Scalar(255, 0, 0), 4);
  if (box.keypoints_size()) {
    for (int i = 0; i < edges.size(); ++i) {
      int idx1 = edges[i].first;
      int idx2 = edges[i].second;
      cv::Point pt1(cols*box.keypoints(idx1*2), rows*box.keypoints(idx1*2+1));
      cv::Point pt2(cols*box.keypoints(idx2*2), rows*box.keypoints(idx2*2+1));
      cv::line(disp, pt1, pt2, cv::Scalar(255, 0, 0), 2);
    }
  }
  return disp;
}


cv::Mat DrawBoxList(const cv::Mat &image, const vlslam_pb::NewBoxList &boxlist) {
  cv::Mat disp(image.clone());
  int rows = image.rows;
  int cols = image.cols;
  for (auto box : boxlist.boxes()) {
    // Not very efficient, but clear ...
    disp = DrawBoxList(disp, box);
  }
  return disp;
}

}

int main(int argc, char **argv) {
    std::string config_file("../cfg/DFTracker.json");
    if (argc > 1) {
        config_file = argv[1];
    }

    auto config = LoadJson(config_file);
    auto cam_cfg = LoadJson(config["camera_config"].asString());

    // setup zmq client
    zmqpp::context context;
    zmqpp::socket socket(context, zmqpp::socket_type::request);
    socket.connect(absl::StrFormat("tcp://localhost:%d", config["port"].asInt()));

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
    cv::namedWindow("Detection", CV_WINDOW_NORMAL);
    cv::Mat disp_tracker, disp_DF, disp_det;

    SE3 camera_pose_t0;

    // initialization in camera frame
    Mat3 Rinit = Mat3::Identity();
    Vec3 Tinit = Vec3::Zero();
    Tinit = GetVectorFromJson<ftype, 3>(config, "Tinit");

    std::shared_ptr<GravityAlignedTracker> tracker{nullptr};

    Timer timer;
    for (int i = 0; i < loader.size(); ++i) {
        cv::Mat img, edgemap;
        vlslam_pb::BoundingBoxList bboxlist;
        SE3 gwc;
        SO3 Rg;

        std::string imagepath;
        bool success = loader.Grab(i, img, edgemap, bboxlist, gwc, Rg, imagepath);
        if (!success) break;

        zmqpp::message msg;
        msg.add_raw<uint8_t>(img.data, img.rows * img.cols * 3);
        socket.send(msg);

        // receive message
        std::string bbox_msg;
        bool recv_ok = socket.receive(bbox_msg);
        if (recv_ok) {
          vlslam_pb::NewBoxList newboxlist;
          newboxlist.ParseFromString(bbox_msg);
          disp_det = DrawBoxList(img, newboxlist);
        } else std::cout << TermColor::red << "failed to receive message" << TermColor::endl;
        absl::SleepFor(absl::Milliseconds(10));

        // std::cout << "gwc=\n" << gwc.matrix3x4() << std::endl;
        // std::cout << "Rg=\n" << Rg.matrix() << std::endl;

        if (tracker == nullptr) {
            tracker = std::make_shared<GravityAlignedTracker>(
                img, edgemap,
                Vec2i{cam_cfg["rows"].asInt(), cam_cfg["cols"].asInt()},
                cam_cfg["fx"].asFloat(), cam_cfg["fy"].asFloat(),
                cam_cfg["cx"].asFloat(), cam_cfg["cy"].asFloat(),
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
        float duration = timer.Tock("tracking");
        std::cout << timer;
        // std::cout << "cost=" << cost << std::endl;
        disp_tracker = tracker->RenderEdgepixels();
        cv::putText(disp_tracker, 
                absl::StrFormat("%0.2f FPS", 1000 / duration),
                cv::Point(20, 20), CV_FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 2);
        cv::imshow("tracker view", disp_tracker);

        disp_DF = tracker->GetDistanceField();
        cv::imshow("DF", disp_DF);

        cv::imshow("Detection", disp_det);

        if (config["save"].asBool()) {
            cv::imwrite(absl::StrFormat("%04d_projection.jpg", i), disp_tracker);
            cv::imwrite(absl::StrFormat("%04d_DF.jpg", i), disp_DF);
        }
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
