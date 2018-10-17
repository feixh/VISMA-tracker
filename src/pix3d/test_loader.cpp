#include <iostream>

#include "glog/logging.h"
#include "renderer.h"
#include "pix3d/dataloader.h"

constexpr float znear = 0.1;
constexpr float zfar = 10;

int main(int argc, char **argv) {
    CHECK_EQ(argc, 2) << "requires root directory of pix3d as an argument!";
    feh::Pix3dLoader loader(argv[1]);
    auto packet = loader.GrabPacket("img/bed/0010.png");
    cv::imshow("image", packet._img);
    cv::imshow("mask", packet._mask);

    std::cout << "object pose=\n" << packet._go.matrix3x4() << std::endl;
    std::cout << "object pose inverse=\n" << packet._go.inverse().matrix3x4() << std::endl;
    std::cout << "camera pose=\n" << packet._gc.matrix3x4() << std::endl;
    std::cout << "camera pose inverse=" << packet._gc.inverse().matrix3x4() << std::endl;
    std::cout << "focal=" << packet._focal_length << std::endl;
    std::cout << "bbox=" << packet._bbox.transpose() << std::endl;
    std::cout << "shape=" << packet._shape.transpose() << std::endl;

    std::cout << packet._V.colwise().mean() << std::endl;
    Eigen::MatrixXf V(packet._V.rows(), packet._V.cols());
    for (int i = 0; i < V.rows(); ++i) {
        auto X = packet._go * packet._V.row(i);
        V.row(i) << -X(0), -X(1), X(2);
    }

    // std::cout << V << std::endl;
    std::cout << "V min=" << V.colwise().minCoeff() << std::endl;
    std::cout << "V max=" << V.colwise().maxCoeff() << std::endl;

    auto engine = std::make_shared<feh::Renderer>(packet._shape[0], packet._shape[1]);
    engine->SetCamera(znear, zfar, packet._focal_length, packet._focal_length, 
            packet._shape[1] >> 1, packet._shape[0] >> 1);
    std::vector<float> VV;
    std::vector<int> FF;
    for (int i = 0; i < V.rows(); ++i) {
        VV.push_back(V(i, 0));
        VV.push_back(V(i, 1));
        VV.push_back(V(i, 2));
    }

    for (int i = 0; i < packet._F.rows(); ++i) {
        FF.push_back(packet._F(i, 0));
        FF.push_back(packet._F(i, 1));
        FF.push_back(packet._F(i, 2));
    }

    engine->SetMesh(VV, FF);
    // render mask
    cv::Mat mask(packet._shape[0], packet._shape[1], CV_8UC1);
    cv::Mat depth(packet._shape[0], packet._shape[1], CV_32FC1);
    mask.setTo(0);
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    // pose(2, 3) = -2.0;
    engine->RenderMask(pose, mask);
    engine->RenderDepth(pose, depth);
    cv::imshow("rendered mask", mask);
    cv::imshow("rendered depth", depth);
    cv::waitKey(0);
    engine.reset();
}
