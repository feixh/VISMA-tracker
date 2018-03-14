//
// Created by feixh on 11/6/17.
//

// stl
#include <iostream>
#include <fstream>

// 3rd party
#include "folly/json.h"
#include "folly/FileUtil.h"
#include "glog/logging.h"
#include "sophus/se3.hpp"

// own
#include "io_utils.h"
#include "tracker_utils.h"
#include "renderer.h"
#include "dataset_loaders.h"
#include "region_based_tracker.h"


int main() {
    std::string config_file("../cfg/region_based_tracker.json");
    std::string content;
    folly::readFile(config_file.c_str(), content);
    folly::dynamic config = folly::parseJson(folly::json::stripComments(content));

    // dataset path
    std::string dataroot = config["dataroot"].asString();
    std::string dataset = config["dataset"].asString();

    // initialize data loader
    feh::LinemodDatasetLoader loader(dataroot + "/" + dataset + "/");

    // initialize renderer
    float z_near = config["camera"]["z_near"].asDouble();
    float z_far = config["camera"]["z_far"].asDouble();
    feh::RendererPtr renderer = std::make_shared<feh::Renderer>(loader.rows_, loader.cols_);
    renderer->SetMesh(loader.vertices(), loader.faces());
    renderer->SetCamera(z_near, z_far, loader.fx_, loader.fy_, loader.cx_, loader.cy_);

    feh::tracker::RegionBasedTracker tracker;
    tracker.Initialize("../cfg/region_based_tracker.json",
                       {loader.fx_, loader.fy_, loader.cx_, loader.cy_, loader.rows_, loader.cols_},
                       loader.vertices(),
                       loader.faces());

    for (int i = 0; i < loader.size(); ++i) {
        cv::Mat img;
        Sophus::SE3f gm_true;
        loader.Grab(i, img, gm_true);

//#define FEH_DEBUG_LINEMOD_DATALOADER
#ifdef  FEH_DEBUG_LINEMOD_DATALOADER
        cv::Mat display = img.clone();
        cv::Mat edge(loader.rows_, loader.cols_, CV_8UC1);
        renderer->RenderEdge(gm_true.matrix(), edge.data);
        feh::tracker::OverlayMaskOnImage(edge, display);
        cv::imshow("input image with gt pose", display);
#else
        Sophus::SE3f gm;
        gm.so3() = Sophus::SO3f::exp(gm_true.so3().log()
                                         + feh::Vec3f::Random() * M_PI * 30 / 180);
        gm.translation() = gm_true.translation() + feh::Vec3f::Random() * 0.01;

        cv::Mat display = img.clone();
        cv::Mat edge(loader.rows_, loader.cols_, CV_8UC1);
        renderer->RenderEdge(gm.matrix(), edge.data);

        uint8_t color[] = {0, 255, 0};
        feh::tracker::OverlayMaskOnImage(edge, display, false, color);

        std::vector<feh::EdgePixel> edgelist;
        renderer->ComputeEdgePixels(gm.matrix(), edgelist);
        cv::Rect bbox = feh::tracker::RectEnclosedByContour(edgelist, loader.rows_, loader.cols_);
        cv::rectangle(display, bbox, cv::Scalar(0, 0, 255), 2);

//        cv::imshow("input image augmented with model", display);
        tracker.Optimize(img, bbox, gm);

//        if (i == 0) {
//            tracker.InitializeTracker(img, bbox, gm);
//        } else {
//            tracker.Update(img);
//        }
#endif


        char ckey = cv::waitKey();
        if (ckey == 'q') break;
    }



}
