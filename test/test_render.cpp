#include "renderer.h"

#include <fstream>
#include "opencv2/opencv.hpp"

#include "io_utils.h"
#include "tracker_utils.h"
#include "common/utils.h"

static const int kRows = 480;
static const int kCols = 640;
static const float kFx = 400;
static const float kFy = 400;
static const float kCx = (kCols >> 1);
static const float kCy = (kRows >> 1);
static const float kZNear = 0.05;
static const float kZFar = 5.0;

float Likelihood_IK(const cv::Mat &prediction, const cv::Mat &evidence) {
    float sum(0);
    for (int i = 0; i < prediction.rows; ++i) {
        for (int j = 0; j < prediction.cols; ++j) {
            sum += std::min(prediction.at<uint8_t>(i, j) / 255.f, evidence.at<uint8_t>(i, j) / 255.f);
        }
    }
    return sum;
}

static const float kEdgeRatio = 0.01;
static const float EPS = 1e-4;
float Likelihood_CE(const cv::Mat &prediction, const cv::Mat &evidence) {
    float sum(0);
    for (int i = 0; i < prediction.rows; ++i) {
        for (int j = 0; j < prediction.cols; ++j) {
            float p = prediction.at<uint8_t>(i, j) / 255.f;
            float e = evidence.at<uint8_t>(i, j) / 255.f;
            sum += (1 - kEdgeRatio) * e * log(EPS + p) + kEdgeRatio * (1 - e) * log(EPS + 1 - p);
        }
    }
    return -sum;
}


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
//    feh::tracker::FlipVertices(v);

    // Initialize OffScreen Renderer
    float intrinsics[] = {kFx, kFy, kCx, kCy};
    feh::Renderer render(kRows, kCols); //, "render1");

    // Set camera and mesh
    render.SetMesh(v, f);
    render.SetCamera(kZNear, kZFar, intrinsics);
//    render.MaxPoolPrediction(3);
    // render.UseGLCoordinateSystem();

    // load edge map
    cv::Mat evidence(kRows, kCols, CV_8UC1);

    feh::Timer timer("renderer");
    Eigen::Matrix4f model;
    model(3, 3) = 1;
    for (int i = 0; i < 72; ++i) {
        model.block<3, 1>(0, 3) = Eigen::Vector3f(0, 0, 1);
        model.block<3, 3>(0, 0) = Eigen::AngleAxisf(2 * M_PI * i / 72, Eigen::Vector3f::UnitY()).toRotationMatrix();
//        std::cout << "model=\n" << model << std::endl;

        std::chrono::high_resolution_clock::time_point t1, t2;

        cv::Mat depthImg(kRows, kCols, CV_32FC1);
//        t1 = std::chrono::high_resolution_clock::now();
        timer.Tick("depth rendering");
        render.RenderDepth(model, (float*)depthImg.data);
        timer.Tock("depth rendering");
//        t2 = std::chrono::high_resolution_clock::now();
//        std::cout << "depth rendering takes " << std::chrono::duration<double, std::milli>(t2 - t1).count()
//                  << " milliseconds" << std::endl;

        cv::namedWindow("depth", CV_WINDOW_NORMAL);
        cv::imshow("depth", depthImg);

#ifdef FEH_RENDER_USE_STENCIL
        // Render boundary image
        cv::Mat boundaryImg(kRows, kCols, CV_8UC1);
        render.RenderBoundaryStencil(model, boundaryImg.data);
        cv::namedWindow("stencil trick", CV_WINDOW_NORMAL);
        cv::imshow("stencil trick", boundaryImg);
#endif

        // Render mask
        cv::Mat mask(kRows, kCols, CV_8UC1);
        mask.setTo(255);
        timer.Tick("mask rendering");
        render.RenderMask(model, mask);
        timer.Tock("mask rendering");
        cv::namedWindow("mask", CV_WINDOW_NORMAL);
        cv::imshow("mask", mask);

        // Render wireframe
        cv::Mat wireframe(kRows, kCols, CV_8UC1);
        timer.Tick("wireframe rendering");
        render.RenderWireframe(model, wireframe);
        timer.Tock("wireframe rendering");
        cv::namedWindow("wireframe", CV_WINDOW_NORMAL);
        cv::imshow("wireframe", wireframe);

        // Render edge
        cv::Mat edgeImg(kRows, kCols, CV_8UC1);
        timer.Tick("edge rendering");
        render.RenderEdge(model, edgeImg.data);
        timer.Tock("edge rendering");
        cv::namedWindow("edge", CV_WINDOW_NORMAL);
        cv::imshow("edge", edgeImg);

        if (i == 0) {
            // flip
            cv::Mat blurred_img(kRows, kCols, CV_8UC1);
            evidence = edgeImg.clone();
            render.UploadEvidence(evidence.data);

//            t1 = std::chrono::high_resolution_clock::now();
//            render.MaxPoolEvidence(3, blurred_img.data);
//            t2 = std::chrono::high_resolution_clock::now();
//            std::cout << "max pooling takes " << std::chrono::duration<float, std::milli>(t2 - t1).count() << " milliseconds" << std::endl;
//            cv::imshow("blurred evidence", blurred_img);
//            cv::waitKey();
        }

        // use intersection kernel to approximate likelihood
        float cpu_time(0);
        std::cout << "==================== Intersection Kernel ===================\n";
        render.UseIntersectionKernel();
        t1 = std::chrono::high_resolution_clock::now();
        float likelihood_cpu_ik = Likelihood_IK(edgeImg, evidence);
        t2 = std::chrono::high_resolution_clock::now();
        cpu_time = std::chrono::duration<float, std::milli>(t2 - t1).count();

        t1 = std::chrono::high_resolution_clock::now();
        float likelihood_gpu_ik = render.Likelihood(model);
        t2 = std::chrono::high_resolution_clock::now();

        std::cout << "GLSL likelihood evaluation takes " << std::chrono::duration<double, std::milli>(t2 - t1).count()
                  << " ms;;; cpu takes " << cpu_time << "ms" << std::endl;
        std::cout << "GLSL likelihood=" << likelihood_gpu_ik << ";;; cpu likelihood=" << likelihood_cpu_ik << std::endl;
//        if (!render.IsEvidenceMaxPooled() && !render.IsPredictionMaxPooled()) {
//            float relative_difference_ik = fabsf(likelihood_gpu_ik - likelihood_cpu_ik) / fabs(likelihood_gpu_ik + likelihood_cpu_ik);
//            CHECK_LE(relative_difference_ik, 0.1f) << "INCONSISTENT Intersection Kernel of CPU & GPU implementation";
//        }

        // use cross entropy to approximate likelihood
        std::cout << "==================== Cross Entropy ===================\n";
        render.UseCrossEntropy();
        t1 = std::chrono::high_resolution_clock::now();
        float likelihood_cpu_ce = Likelihood_CE(edgeImg, evidence);
        t2 = std::chrono::high_resolution_clock::now();
        cpu_time = std::chrono::duration<float, std::milli>(t2 - t1).count();

        t1 = std::chrono::high_resolution_clock::now();
        float likelihood_gpu_ce = render.Likelihood(model);
        t2 = std::chrono::high_resolution_clock::now();

        std::cout << "GLSL likelihood evaluation takes " << std::chrono::duration<double, std::milli>(t2 - t1).count()
                  << " ms;;; cpu takes " << cpu_time << "ms" << std::endl;
        std::cout << "GLSL likelihood=" << likelihood_gpu_ce << ";;; cpu likelihood=" << likelihood_cpu_ce << std::endl;
//        if (!render.IsEvidenceMaxPooled() && !render.IsPredictionMaxPooled()) {
//            float relative_difference_ce = fabsf(likelihood_gpu_ce - likelihood_cpu_ce) / fabs(likelihood_gpu_ce + likelihood_cpu_ce);
//            CHECK_LE(relative_difference_ce, 0.1f) << "INCONSISTENT CrossEntropy of CPU & GPU implementation";
//        }
        std::cout << "==================== End ===================\n";

#ifdef FEH_RENDER_USE_CUDA

        t1 = std::chrono::high_resolution_clock::now();
        float value = render.Likelihood(model);
        t2 = std::chrono::high_resolution_clock::now();
        std::cout << "cuda likelihood evaluation takes " << std::chrono::duration<double, std::milli>(t2 - t1).count()
                  << " ms;;; cpu takes " << cpu_time << "ms" << std::endl;
        std::cout << "cuda likelihood=" << value << ";;; cpu likelihood=" << sum << std::endl;
#endif

        char c = cv::waitKey(200);
        if (c == 'q') break;
        std::cout << timer;
    }

}
