//
// Created by visionlab on 1/27/18.
//
#pragma once
#include <iostream>
#include "vlslam.pb.h"

namespace feh {

class BBoxLikelihoodHandler {
public:
    void Handle(const lcm::ReceiveBuffer* rawbuf,
                const std::string& channel) {
        scores_.clear();
        vlslam_pb::BoundingBoxList bboxlist;
        bboxlist.ParseFromArray(rawbuf->data, rawbuf->data_size);
        tracker_id_ = atoi(bboxlist.description().substr(0, 4).c_str());
//        std::ofstream out("rec.txt", std::ios::out);
//        bboxlist.SerializeToOstream(&out);
//        out.close();
        for (const auto &bbox : bboxlist.bounding_boxes()) {
//            // (x1, y1)-(x2, y2): score
//            std::cout << "(" << bbox.top_left_x() << "," << bbox.top_left_y() << ")"
//                      << "-"
//                      << "(" << bbox.bottom_right_x() << "," << bbox.bottom_right_y() << ")"
//                      << ":"
//                      << bbox.scores(0) << "\n";
            scores_.push_back(bbox.scores(0));
        }
    }
    std::vector<float> scores_;
    uint32_t tracker_id_;
};

}
