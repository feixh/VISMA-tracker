//
// Created by visionlab on 2/12/18.
//
// Multiple OpenGL-based renderer in one thread.
#include "renderer.h"

// stl
#include <fstream>

// 3rd party
#include "opencv2/opencv.hpp"
#include "tbb/parallel_for.h"

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

    std::vector<feh::RendererPtr> renderers;
    for (int i = 0; i < 10; ++i) {
        feh::RendererPtr render = std::make_shared<feh::Renderer>(kRows, kCols);
        // Set camera and mesh
        render->SetCamera(kZNear, kZFar, intrinsics);
        render->SetMesh(v, f);
        renderers.push_back(render);
    }

    Eigen::Matrix4f model;
    model(3, 3) = 1.0f;
    model.block<3, 1>(0, 3) = Eigen::Vector3f(0, 0, 1);
    model.block<3, 3>(0, 0) = Eigen::AngleAxisf(M_PI * 0.5, Eigen::Vector3f::UnitY()).toRotationMatrix();
    std::vector<Eigen::Matrix4f> models(1000, model);

//    for (auto r : renderers) {
//        cv::Mat wireframe(kRows, kCols, CV_8UC1);
//        r->RenderWireframe(model, wireframe);
//        cv::imshow("wireframe"+ r->id(), wireframe);
//        char ckey = cv::waitKey();
//    }

    feh::Timer timer;
    timer.Tick("multi-render");
    auto op = [kRows, kCols, &renderers, &models] (const tbb::blocked_range<int> &range) {
        for (int i = range.begin(); i < range.end(); ++i) {
            int rid = i / range.grainsize();
            CHECK_LT(rid, renderers.size());
            cv::Mat wireframe(kRows, kCols, CV_8UC1);
            renderers[rid]->RenderWireframe(models[i], wireframe);
//            char ss[256];
//            sprintf(ss, "../dump/wireframe_%04d_r%04d.png", i, rid);
//            cv::imwrite(ss, wireframe);
        }
    };
    tbb::parallel_for(
        tbb::blocked_range<int>(0, models.size(), models.size()/renderers.size()),
        op);
    timer.Tock("multi-render");

    timer.Tick("single-render");
    for (int i = 0; i < models.size(); ++i) {
        cv::Mat wireframe(kRows, kCols, CV_8UC1);
        renderers[0]->RenderWireframe(models[i], wireframe);
//        char ss[256];
//        sprintf(ss, "../dump/wireframe_%04d_r%04d.png", i, rid);
//        cv::imwrite(ss, wireframe);
    }
    timer.Tock("single-render");


    std::cout << timer;

//    cv::Mat image(kRows, kCols, CV_8UC3);
//    image.setTo(255);
//    uint8_t color[] = {255, 0, 0};
//    feh::tracker::OverlayMaskOnImage(wireframe, image, true, color);
//    cv::imshow("wireframe", image);
//    cv::waitKey();

}

