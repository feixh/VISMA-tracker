//
// Created by visionlab on 10/17/17.
//

#include "tracker.h"
#include "tracker_utils.h"
#include <chrono>

int main() {
    feh::tracker::Tracker tracker;
    tracker.Initialize("../cfg/chair_tracker.json");
    std::cout << "switch context\n";

    feh::Mat4f model_pose = feh::tracker::Mat4FromState({0, 0.0, 0.8, M_PI * -1.5});

    cv::Mat edge(tracker.renderer().rows(), tracker.renderer().cols(), CV_8UC1);
    float total(0);
    for (int i = 0; i < 100; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        tracker.renderer().RenderEdge(Sophus::SE3f(model_pose).matrix(), edge.data);
        auto t2 = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<float, std::milli>(t2 - t1).count();
    }
    std::cout << "edge rendering takes " << total / 100.f << " ms on average\n";

    cv::imshow("edge", edge);
    cv::waitKey();
}
