#include <iostream>

#include "glog/logging.h"
#include "fmt/format.h"

#include "renderer.h"
#include "pix3d/dataloader.h"
#include "distance_transform.h"

constexpr float znear = 0.1;
constexpr float zfar = 10;

int main(int argc, char **argv) {
    CHECK_EQ(argc, 2) << "requires root directory of pix3d as an argument!";
    feh::Pix3dLoader loader(argv[1]);
    // auto packet = loader.GrabPacket("img/bed/0010.png"); // index by path
    // OR index by id
    auto packet = loader.GrabPacket(0); // index by path

    cv::namedWindow("image", CV_WINDOW_NORMAL);
    cv::imshow("image", packet.img_);

    cv::namedWindow("mask", CV_WINDOW_NORMAL);
    cv::imshow("mask", packet.mask_);

    cv::namedWindow("edge", CV_WINDOW_NORMAL);
    cv::imshow("edge", packet.edge_);

    cv::Mat dt;
    cv::Mat normalized_edge;
    normalized_edge = cv::Scalar::all(255) - packet.edge_;
    cv::distanceTransform(normalized_edge / 255.0, dt, CV_DIST_L2, CV_DIST_MASK_PRECISE);
    cv::namedWindow("dt", CV_WINDOW_NORMAL);
    cv::Mat dt_to_show = feh::DistanceTransform::BuildView(dt);
    cv::imshow("dt", dt_to_show);

    std::cout << "object pose=\n" << packet.go_.matrix3x4() << std::endl;
    std::cout << "object pose inverse=\n" << packet.go_.inverse().matrix3x4() << std::endl;
    std::cout << "camera pose=\n" << packet.gc_.matrix3x4() << std::endl;
    std::cout << "camera pose inverse=" << packet.gc_.inverse().matrix3x4() << std::endl;
    std::cout << "focal=" << packet.focal_length_ << std::endl;
    std::cout << "bbox=" << packet.bbox_.transpose() << std::endl;
    std::cout << "shape=" << packet.shape_.transpose() << std::endl;

    std::cout << packet.V_.colwise().mean() << std::endl;
    Eigen::MatrixXf V(packet.V_.rows(), packet.V_.cols());
    for (int i = 0; i < V.rows(); ++i) {
        auto X = packet.go_ * packet.V_.row(i);
        V.row(i) << -X(0), -X(1), X(2);
    }

    // std::cout << V << std::endl;
    std::cout << "V min=" << V.colwise().minCoeff() << std::endl;
    std::cout << "V max=" << V.colwise().maxCoeff() << std::endl;

    auto engine = std::make_shared<feh::Renderer>(packet.shape_[0], packet.shape_[1]);
    engine->SetCamera(znear, zfar, packet.focal_length_, packet.focal_length_,
            packet.shape_[1] >> 1, packet.shape_[0] >> 1);
    std::vector<float> VV;
    std::vector<int> FF;
    for (int i = 0; i < V.rows(); ++i) {
        VV.push_back(V(i, 0));
        VV.push_back(V(i, 1));
        VV.push_back(V(i, 2));
    }

    for (int i = 0; i < packet.F_.rows(); ++i) {
        FF.push_back(packet.F_(i, 0));
        FF.push_back(packet.F_(i, 1));
        FF.push_back(packet.F_(i, 2));
    }

    engine->SetMesh(VV, FF);
    // render mask
    cv::Mat mask(packet.shape_[0], packet.shape_[1], CV_8UC1);
    cv::Mat depth(packet.shape_[0], packet.shape_[1], CV_32FC1);
    mask.setTo(0);
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    // pose(2, 3) = -2.0;
    engine->RenderMask(pose, mask);
    engine->RenderDepth(pose, depth);

    cv::namedWindow("rendered mask", CV_WINDOW_NORMAL);
    cv::imshow("rendered mask", mask);

    cv::namedWindow("rendered depth", CV_WINDOW_NORMAL);
    cv::imshow("rendered depth", depth);
    engine.reset();

    // now let's compare the rendered mask against the provided mask
    cv::Mat mask_diff(packet.shape_[0], packet.shape_[1], CV_8UC1);
    mask_diff.setTo(0);
    for (int i = 0; i < packet.shape_[0]; ++i) {
        for (int j = 0; j < packet.shape_[1]; ++j) {
            if ( ((int)packet.mask_.at<uint8_t>(i, j)+ (int)mask.at<uint8_t>(i, j)) != 255) {
                std::cout << fmt::format("({}, {}) not consistent", i, j) << std::endl;
                mask_diff.at<uint8_t>(i, j) = 255;
            }
        }
    }
    cv::namedWindow("mask difference", CV_WINDOW_NORMAL);
    cv::imshow("mask difference", mask_diff);
    cv::waitKey();
}
