//
// Created by feixh on 10/25/17.
//
#include "distance_transform.h"
#include "tracker_utils.h"

int main() {
    std::vector<float> f = {10, 10, 10, 0, 10, 1, 12, 3, 0, 1, 1, 1, 0};
    feh::DistanceTransform dt;
    std::vector<float> d;
    dt(f, d);

    for (int i = 0; i < f.size(); ++i) {
        std::cout << f[i] << " ";
    }
    std::cout << "\n";

    for (int i = 0; i < d.size(); ++i) {
        std::cout << d[i] << " ";
    }
    std::cout << "\n";

    cv::Mat edgemap = cv::imread("../resources/edgemap.png", 0);
    CHECK(edgemap.type() == CV_8UC1);
    cv::Mat im = feh::DistanceTransform::Preprocess(edgemap);

    feh::Timer timer;
    cv::Mat dt_im;
    timer.Tick("dt");
    dt(im, dt_im);
    timer.Tock("dt");

    cv::Mat dt_im2;
    timer.Tick("dt");
    dt(im / 200.0f, dt_im2);
    timer.Tock("dt");

    std::cout << "abs sum of difference=" << cv::sum(cv::abs(dt_im2 - dt_im)) << "\n";
    timer.report_average = true;
    std::cout << timer;

    cv::Mat display = feh::DistanceTransform::BuildView(dt_im);

    cv::imshow("in image", edgemap);
    cv::imshow("dt image", display);
    cv::waitKey();


}

