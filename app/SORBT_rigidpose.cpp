//
// Created by feixh on 11/16/17.
//
#include "dataloaders.h"

// stl
#include <fstream>

// own
#include "renderer.h"
#include "tracker_utils.h"
#include "region_based_tracker.h"

using namespace feh;

using Tag = RigidPoseDatasetLoader::Tag;

int main() {
    auto config = LoadJson("../cfg/rigidpose.json");

    int tag = (config["noise_level"].asInt() << 4)
              | config["left_right"].asInt();
    RigidPoseDatasetLoader loader(config["dataroot"].asString(),
                                       config["dataset"].asString(),
                                       tag);

    // tracker
    tracker::RegionBasedTracker tracker;
    tracker.Initialize("../cfg/rigidpose.json",
                       {loader.focal_length_, loader.focal_length_,
                       loader.cx_, loader.cy_, loader.rows_, loader.cols_},
                       loader.vertices(),
                       loader.faces());

    // tmp
    Renderer renderer(loader.rows_, loader.cols_);
    renderer.SetCamera(0.05, 5.0,
                       loader.focal_length_, loader.focal_length_,
                       loader.cx_, loader.cy_);
    renderer.SetMesh(loader.vertices(), loader.faces());

    cv::Mat image;
    cv::Mat edge(loader.rows_, loader.cols_, CV_8UC1);
    SE3 pose;
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
//        tracker::OverlayMaskOnImage(edge, display, true, tracker::kColorGreen);

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
            std::vector<EdgePixel> edgelist;
            renderer.ComputeEdgePixels(pose.matrix(), edgelist);
            cv::Rect bbox = tracker::RectEnclosedByContour(
                edgelist, loader.rows_, loader.cols_);

            bbox = tracker::InflateRect(bbox, loader.rows_, loader.cols_, 20);

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


