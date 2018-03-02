//
// Created by visionlab on 11/2/17.
//

#include "region_based_tracker.h"

int main() {
    cv::Mat image;
    image = cv::imread("../resources/leather_chair.png");
    cv::Rect bbox(cv::Point(305, 174), cv::Point(566, 451));
//    image = cv::imread("../resources/swivel_chair.png");
//    cv::Rect bbox(cv::Point(400, 150), cv::Point(700, 500));

    cv::Mat display(image.clone());
    cv::rectangle(display, bbox, cv::Scalar(255, 0, 0));
    cv::imshow("image and bbox", display);
    cv::waitKey();

    feh::tracker::RegionBasedTracker tracker;
    tracker.Initialize("../cfg/region_based_tracker.json");
    tracker.Optimize(image, bbox);
}

