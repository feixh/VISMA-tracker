#include <iostream>
#include <tuple>
#include <chrono>

#include "glog/logging.h"
#include "folly/Format.h"
#include "opencv2/imgproc.hpp"

#include "eigen_alias.h"
#include "utils.h"
#include "rodrigues.h"
#include "distance_transform.h"
#include "renderer.h"
#include "tracker_utils.h"
#include "pix3d/dataloader.h"
#include "pix3d/diff_tracker.h"

int main(int argc, char **argv) {
    // CHECK_EQ(argc, 2) << "requires root directory of pix3d as an argument!";
    feh::Pix3dLoader loader(argv[1]);
    // auto packet = loader.GrabPacket("img/bed/0010.png"); // index by path
    // OR index by id
    auto packet = loader.GrabPacket(0); // index by path

    cv::namedWindow("image", CV_WINDOW_NORMAL);
    cv::imshow("image", packet._img);

    cv::namedWindow("mask", CV_WINDOW_NORMAL);
    cv::imshow("mask", packet._mask);

    cv::namedWindow("edge", CV_WINDOW_NORMAL);
    
    cv::imshow("edge", packet._edge);

    // noise generators
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto generator = std::make_shared<std::knuth_b>(seed);
    std::normal_distribution<float> normal_dist;
    float Tnoise = std::stof(argv[2]);
    float Rnoise = std::stof(argv[3]);

    feh::Vec3 Tn = packet._go.translation() 
        + Tnoise * feh::RandomVector<3>(0, Tnoise, generator);

    feh::Vec3 Wn = packet._go.so3().log() 
        + Rnoise * feh::RandomVector<3>(0, Rnoise, generator);

    feh::Mat3 Rn = rodrigues(Wn);


    feh::DiffTracker tracker(packet._img, packet._edge,
            packet._shape, 
            packet._focal_length, packet._focal_length,
            packet._shape[1] >> 1, packet._shape[0] >> 1,
            Rn, Tn,
            packet._V, packet._F);
    cv::namedWindow("depth", CV_WINDOW_NORMAL);
    cv::namedWindow("edgepixels", CV_WINDOW_NORMAL);
    cv::namedWindow("DF", CV_WINDOW_NORMAL);
    cv::namedWindow("dDF_dx", CV_WINDOW_NORMAL);
    cv::namedWindow("dDF_dy", CV_WINDOW_NORMAL);

    cv::imshow("DF", tracker.GetDistanceField());

    feh::Mat3 flip;
    flip << -1, 0, 0,
         0, -1, 0,
         0, 0, 1;

    feh::Timer timer;
    for (int i = 0; i < 100; ++i) {
        std::cout << "==========\n";
        timer.Tick("update");
        auto cost = tracker.Minimize(1);
        timer.Tock("update");
        std::cout << "Iter=" << i << std::endl;
        std::cout << "Cost=" << cost << std::endl;

        feh::Mat3 Re;
        feh::Vec3 Te;
        std::tie(Re, Te) = tracker.GetEstimate();
        Re = flip * Re;
        Te = flip * Te; // flip back
        Eigen::Matrix<float, 6, 1> err;
        err << invrodrigues(feh::Mat3{Re.transpose() * packet._go.so3().matrix()}),
            Te - packet._go.translation();
        std::cout << "Error vector=" << err.transpose() << std::endl;
        std::cout << "R Error=" << err.head<3>().norm() / 3.14 * 180 << std::endl;
        std::cout << "T Error=" << 100 * err.tail<3>().norm() / (packet._V.maxCoeff() - packet._V.minCoeff())<< std::endl;

        cv::Mat depth = tracker.RenderEstimate();
        cv::imshow("depth", depth);
        cv::Mat overlaid_view = tracker.RenderEdgepixels();
        cv::imshow("edgepixels", overlaid_view);
        cv::imwrite(folly::sformat("{:04d}.jpg", i), overlaid_view);
        cv::imshow("dDF_dx", std::get<0>(tracker.GetDFGradient()));
        cv::imshow("dDF_dy", std::get<1>(tracker.GetDFGradient()));


        char ckey = cv::waitKey(10);
        if (ckey == 'q') break;
    }
    std::cout << timer;

}

