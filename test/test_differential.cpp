//
// Created by feixh on 10/29/17.
//
#include "renderer.h"

// stl
#include <fstream>

// 3rd party
#include "opencv2/opencv.hpp"

// own
#include "io_utils.h"
#include "tracker_utils.h"

static const int kRows = 480;
static const int kCols = 640;
static const float kFx = 400;
static const float kFy = 400;
static const float kCx = (kCols >> 1);
static const float kCy = (kRows >> 1);
static const float kZNear = 0.05;
static const float kZFar = 5.0;


int main(int argc, char **argv) {

    std::string obj_file_path("../resources/swivel_chair.obj");
//    std::string obj_file_path("/mnt/external/Dropbox/CVPR18/data/CAD_database/swivel_chair.obj");
    if (argc == 2) {
        obj_file_path = std::string(argv[1]);
    } else if (argc != 1) {
        LOG(FATAL) << "invalid argument format";
    }

    std::vector<float> v;
    std::vector<int> f;
    feh::io::LoadMeshFromObjFile(obj_file_path, v, f);
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
    model.block<3, 3>(0, 0) = Eigen::AngleAxisf(M_PI * 0.5, Eigen::Vector3f::UnitY()).toRotationMatrix();
    feh::Timer timer;
    std::vector<feh::EdgePixel> edgelist;
    for (int i = 0; i < 100; ++i) {
        timer.Tick("construct edgelist");
        render.ComputeEdgePixels(model, edgelist);
        timer.Tock("construct edgelist");
    }
    timer.report_average = true;

    std::vector<float> pointcloud;
    for (const auto &edgepixel : edgelist) {
        feh::Vec3f xc((edgepixel.x - kCx) / kFx,
                     (edgepixel.y - kCy) / kFy,
                     1.0);
        xc *= edgepixel.depth;
        for (int i = 0; i < 3; ++i) pointcloud.push_back(xc(i));
    }

    cv::Mat depth(kRows, kCols, CV_32FC1);
    render.RenderDepth(model, (float*)depth.data);
    cv::imshow("depth", depth);
    cv::waitKey();

//    std::ofstream out("boundary_pointcloud.ply", std::ios::out | std::ios::binary);
//    CHECK(out.is_open());
//    {
//        tinyply::PlyFile plyfile;
//        plyfile.add_properties_to_element("vertex", {"x", "y", "z"}, pointcloud);
//        plyfile.write(out, true);
//    }
//    out.close();

    pointcloud.clear();
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < kCols; ++j) {
            float zb = depth.at<float>(i, j);
            if (zb < 1) {
                feh::Vec3f pt((i - kCx) / kFx, (j - kCy) / kFy, 1.0);
                float z = feh::LinearizeDepth(zb, kZNear , kZFar);
                pt *= z;
                for (int k = 0; k < 3; ++k) {
                    pointcloud.push_back(pt(k));
                }
            }
        }
    }
//    out.open("pointcloud.ply", std::ios::out | std::ios::binary);
//    CHECK(out.is_open());
//    {
//        tinyply::PlyFile plyfile;
//        plyfile.add_properties_to_element("vertex", {"x", "y", "z"}, pointcloud);
//        plyfile.write(out, true);
//    }
//    out.close();


}

