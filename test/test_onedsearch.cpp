#include "renderer.h"

#include <fstream>

#include "opencv2/opencv.hpp"

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
    cv::Mat target_edge(kRows, kCols, CV_8UC1);
    render.RenderEdge(model, target_edge.data);

    Eigen::Matrix4f model_init(model);
    model_init(0, 3) += 0.0;    // x
    model_init(1, 3) += -0.01;    // y
    model_init(2, 3) += 0.21;    // z
    cv::Mat init_edge(kRows, kCols, CV_8UC1);
    render.RenderEdge(model_init, init_edge.data);

    // build the list of edge pixels
    feh::Timer timer;
    std::vector<feh::EdgePixel> edgelist;
    for (int i = 0; i < 100; ++i) {
        timer.Tick("construct edgelist");
        render.ComputeEdgePixels(model_init, edgelist);
        timer.Tock("construct edgelist");
    }
    timer.report_average = true;


    // visualize search direction
    cv::Mat normal_view(kRows, kCols, CV_8UC3);
    cv::Mat normal_view2(kRows, kCols, CV_8UC3);
    feh::OneDimSearch::BuildNormalView(edgelist, normal_view);
    feh::OneDimSearch::BuildNormalView2(edgelist, normal_view2);

    // one dimensional search along normal of edge pixels
    std::vector<feh::OneDimSearchMatch> matches;
    feh::OneDimSearch search;
    search.step_length_ = 1;
    search.direction_consistency_thresh_ = 1.9;
//    search(edgelist, target_edge, matches, target_dir);
    search(edgelist, target_edge, matches);

    std::vector<feh::OneDimSearchMatch> parallel_matches;
    search.Parallel(edgelist, target_edge, parallel_matches);

    cv::namedWindow("match view", CV_WINDOW_NORMAL);
    cv::Mat match_view;
    feh::OneDimSearch::BuildMatchView(normal_view, target_edge, matches, match_view);

    cv::imshow("match view", match_view);
    cv::imshow("normal view", normal_view);
    cv::namedWindow("normal view2", CV_WINDOW_NORMAL);
    cv::imshow("normal view2", normal_view2);
    cv::imshow("initial edgemap", init_edge);
    cv::waitKey();

    // timing
    // enlarge edgelist by appending itself
    int n = edgelist.size();
    std::vector<feh::EdgePixel> biglist(n * 10);
    for (int i = 0; i < 10; ++i) {
        std::copy(edgelist.begin(), edgelist.end(), biglist.begin() + i * n);
    }
    timer.Tick("1-d search");
    search(biglist, target_edge, matches);
    timer.Tock("1-d search");

    timer.Tick("parallel 1-d search");
    search(biglist, target_edge, parallel_matches);
    timer.Tock("parallel 1-d search");
    std::cout << timer;
    CHECK_EQ(matches.size(), parallel_matches.size());




}

