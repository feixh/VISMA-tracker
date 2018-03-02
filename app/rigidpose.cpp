//
// Created by feixh on 11/16/17.
//
#include "dataset_loaders.h"

// stl
#include <fstream>

// 3rd party
#include "json/json.h"
#include "glog/logging.h"

// own
#include "renderer.h"
#include "tracker_utils.h"
#include "region_based_tracker.h"

using Tag = feh::RigidPoseDatasetLoader::Tag;

int main() {

    std::ifstream ifs("../cfg/rigidpose.json", std::ios::in);
    CHECK(ifs.is_open()) << "failed to open configure file";

    Json::Value config;
    ifs >> config;
    ifs.close();

    int tag = (config["noise_level"].asInt() << 4)
              | config["left_right"].asInt();
    feh::RigidPoseDatasetLoader loader(config["dataroot"].asString(),
                                       config["dataset"].asString(),
                                       tag);

    // tracker
    feh::tracker::RegionBasedTracker tracker;
    tracker.Initialize("../cfg/rigidpose.json",
                       {loader.focal_length_, loader.focal_length_,
                       loader.cx_, loader.cy_, loader.rows_, loader.cols_},
                       loader.vertices(),
                       loader.faces());

    // tmp
    feh::Renderer renderer(loader.rows_, loader.cols_);
    renderer.SetCamera(0.05, 5.0,
                       loader.focal_length_, loader.focal_length_,
                       loader.cx_, loader.cy_);
    renderer.SetMesh(loader.vertices(), loader.faces());

    cv::Mat image;
    cv::Mat edge(loader.rows_, loader.cols_, CV_8UC1);
    Sophus::SE3f pose;
    int frame_counter(0);
    while (loader.Grab(image, pose)) {
        cv::flip(image, image, 0);

        cv::Mat display;
        display = image.clone();

//#define FEH_DEBUG_RIGIDPOSE_DATALOADER
#ifdef  FEH_DEBUG_RIGIDPOSE_DATALOADER
//        // debugging dataset loader
//        renderer.SetCamera(pose.matrix());
////        renderer.RenderWireframe(pose.matrix(), edge.data);
//        renderer.RenderWireframe(Eigen::Matrix4f::Identity(), edge.data);
//        feh::tracker::OverlayMaskOnImage(edge, display, true, feh::tracker::kColorGreen);

        auto v = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>(&loader.vertices()[0], 3, loader.vertices().size()/3);
        for (int i = 0; i < v.cols(); ++i) {
            auto vc = pose * v.block<3, 1>(0, i);
            float x = loader.focal_length_ * vc(0) / vc(2) + loader.cx_;
            float y = loader.focal_length_ * vc(1) / vc(2) + loader.cy_;
            cv::circle(display, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
        }

        cv::imshow("input image", display);
#else

        if (frame_counter == 0) {
            std::vector<feh::EdgePixel> edgelist;
            renderer.ComputeEdgePixels(pose.matrix(), edgelist);
            cv::Rect bbox = feh::tracker::RectEnclosedByContour(
                edgelist, loader.rows_, loader.cols_);

            bbox = feh::tracker::InflateRect(bbox, loader.rows_, loader.cols_, 20);

//            tracker.Optimize(image, bbox, pose);
            tracker.InitializeTracker(image, bbox, pose);
        } else {
            tracker.Update(image);
        }
#endif
        char ckey = cv::waitKey(24);
        if (ckey == 'q') break;

        ++frame_counter;
    }

    cv::destroyAllWindows();


}


