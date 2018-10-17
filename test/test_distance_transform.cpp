//
// Created by visionlab on 10/24/17.
//
#include "renderer.h"

#include <fstream>
#include <chrono>

#include "opencv2/opencv.hpp"

#include "utils.h"
#include "tracker_utils.h"
#include "oned_search.h"
#include "distance_transform.h"

static const int kRows = 480;
static const int kCols = 640;
static const float kFx = 400;
static const float kFy = 400;
static const float kCx = (kCols >> 1);
static const float kCy = (kRows >> 1);
static const float kZNear = 0.05;
static const float kZFar = 5.0;


int main(int argc, char **argv) {

    std::string obj_file_path(argv[1]);
//    std::string obj_file_path("/mnt/external/Dropbox/CVPR18/data/CAD_database/swivel_chair.obj");
    if (argc == 2) {
        obj_file_path = std::string(argv[1]);
    } else if (argc != 1) {
        LOG(FATAL) << "invalid argument format";
    }

    std::vector<float> v;
    std::vector<int> f;
    feh::LoadMeshFromObjFile(obj_file_path, v, f);
    feh::tracker::NormalizeVertices(v);
    feh::tracker::FlipVertices(v);
    // Initialize OffScreen Renderer
    float intrinsics[] = {kFx, kFy, kCx, kCy};
    feh::Renderer render(kRows, kCols); //, "render1");

    // Set camera and mesh
    render.SetMesh(v, f);
    render.SetCamera(kZNear, kZFar, intrinsics);
    render.SetMesh(v, f);

    Eigen::Matrix4f model;
    model(3, 3) = 1.0f;
    model.block<3, 1>(0, 3) = Eigen::Vector3f(0, 0, 1);
    model.block<3, 3>(0, 0) = Eigen::AngleAxisf(-M_PI * 0.6, Eigen::Vector3f::UnitY()).toRotationMatrix();
    cv::Mat target_edge(kRows, kCols, CV_8UC1);
    render.RenderEdge(model, target_edge.data);

    // flip intensity values
    // since distance transform is supposed to compute
    // D(p) = min(d(p,q) + f(q)), given pixel location p
    // We can also put weight alpha on the second term, i.e., alpha f(q)
    // which is equivalent to scale f(q) before-hand.
    // reference:
    // http://www.cs.cornell.edu/~dph/papers%5Cdt.pdf
    cv::Mat normalized_edge;
    normalized_edge = cv::Scalar::all(255) - target_edge;

    cv::Mat dt;
    feh::Timer timer;
    timer.Tick("distance transform");
    cv::distanceTransform(normalized_edge, dt, CV_DIST_L2, CV_DIST_MASK_PRECISE);
    timer.Tock("distance transform");
    timer.report_average = true;
    CHECK_EQ(dt.type(), CV_32F);


    // 2nd time distance transform with scaled input
    cv::Mat dt2;
    cv::distanceTransform(normalized_edge / 255.0f, dt2, CV_DIST_L2, CV_DIST_MASK_PRECISE);

    std::cout << "abs sum of difference=" << cv::sum(cv::abs(dt2-dt)) << "\n";

    cv::Mat display = feh::DistanceTransform::BuildView(dt);


    // my own distance transform
    cv::Mat mydt_in = feh::DistanceTransform::Preprocess(target_edge);

    feh::DistanceTransform mydt_func;
    cv::Mat mydt;
    timer.Tick("mydt");
    mydt_func(mydt_in * 5, mydt);
    timer.Tock("mydt");
    cv::Mat display2 = feh::DistanceTransform::BuildView(mydt);

    std::cout << timer;


    while (1) {
        cv::imshow("edge map", target_edge);
        cv::imshow("raw distance field", dt);
        cv::imshow("scaled distance field", display);
        cv::imshow("scaled distance field2", display2);
        char c = cv::waitKey(30);
        if (c == 'q') break;
    }


}


