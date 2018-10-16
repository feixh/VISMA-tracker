#include <iostream>

#include "pix3d/dataloader.h"

int main() {
    feh::Pix3dLoader loader("/home/visionlab/Data/pix3d");
    auto packet = loader.GrabPacket(0);
    cv::imshow("image", packet._img);
    cv::imshow("mask", packet._mask);
    cv::waitKey(0);

    std::cout << "pose=\n" << packet._g.matrix3x4() << std::endl;
    std::cout << "cam position=" << packet._cam_position.transpose() << std::endl;
    std::cout << "focal=" << packet._focal_length << std::endl;
    std::cout << "inplaine rotation=" << packet._inplane_rotation << std::endl;
    std::cout << "bbox=" << packet._bbox.transpose() << std::endl;

    // TODO: sanity check -- render the model at the given pose and compare to RGB image
}
