//
// Created by feixh on 11/7/17.
//
#include "undistorter.h"

int main() {
    cv::Mat image = cv::imread("../resources/leather_chair.png");
    feh::UndistorterPTAM undistorter(0.561859, 0.90154, 0.491896, 0.512629, 0.709402,
                                     500, 960,
                                     "crop",
                                     500, 960);
    cv::Mat undistorted_image;
    cv::Mat gray;
    cv::cvtColor(image, gray, CV_RGB2GRAY);
    undistorter.undistort(gray, undistorted_image);
    cv::imshow("undistorted", undistorted_image);
    cv::waitKey();
}

