//
// Created by visionlab on 11/6/17.
//
#include "renderer.h"

// stl
#include <fstream>

// 3rd party
#include "opencv2/opencv.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

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

    std::string obj_file_path("../resources/swivel_chair_scanned.obj");
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
    render.SetCamera(kZNear, kZFar, intrinsics);
    render.SetMesh(v, f);

    Eigen::Matrix4f model;
    model(3, 3) = 1.0f;
    model.block<3, 1>(0, 3) = Eigen::Vector3f(0, 0, 1);
    model.block<3, 3>(0, 0) = Eigen::AngleAxisf(M_PI * 0.5, Eigen::Vector3f::UnitY()).toRotationMatrix();

    cv::Mat wireframe(kRows, kCols, CV_8UC1);
    render.RenderWireframe(model, wireframe);
    cv::imshow("wireframe", wireframe);
    cv::waitKey();

    cv::Mat image(kRows, kCols, CV_8UC3);
    image.setTo(255);
    uint8_t color[] = {255, 0, 0};
    feh::tracker::OverlayMaskOnImage(wireframe, image, true, color);
    cv::imshow("wireframe", image);
    cv::waitKey();
}

