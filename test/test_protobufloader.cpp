//
// Created by feixh on 10/17/17.
//
#include <fstream>
#include <iostream>
#include <exception>

#include "glog/logging.h"
#include "opencv2/opencv.hpp"


// protobuf
#include "vlslam.pb.h"

// feh
#include "io_utils.h"

int main() {
    std::string dataset_root("/local/feixh/tmp/swivel_chair/");
    vlslam_pb::Dataset dataset;
    std::ifstream in_file(dataset_root + "dataset");
    if (!in_file.is_open()) {
        LOG(FATAL) << "FATAL::failed to open dataset";
    }
    dataset.ParseFromIstream(&in_file);
    in_file.close();

    std::vector<std::string> png_files;
    if (!feh::io::Glob(dataset_root, ".png", png_files)) {
        LOG(FATAL) << "FATAL::failed to read png file list @" << dataset_root;
    }

    std::vector<std::string> edge_files;
    if (!feh::io::Glob(dataset_root, ".edge", edge_files)) {
        LOG(FATAL) << "FATAL::failed to read edge map list @" << dataset_root;
    }

    std::vector<std::string> bbox_files;
    if (!feh::io::Glob(dataset_root, ".bbox", bbox_files)) {
        LOG(FATAL) << "FATAL::failed to read bounding box lisst @" << dataset_root;
    }

    for (int i = 0; i < png_files.size(); ++i) {
        std::string png_file = png_files[i];
        std::string edge_file = edge_files[i];
        std::string bbox_file = bbox_files[i];

        // read image
        cv::Mat img = cv::imread(png_file);

        // read edgemap
        cv::Mat edgemap;
        if (feh::io::LoadEdgeMap(edge_file, edgemap)) {
            cv::imshow("edge", edgemap);
        } else {
            LOG(FATAL) << "failed to load edge map @ " << edge_file;
        }

        // read bounding box
        vlslam_pb::BoundingBoxList bboxlist;
        in_file.open(bbox_file);
        if (!in_file.is_open()) {
            LOG(FATAL) << "FATAL::failed to open bbox file @ " << bbox_file;
        }
        bboxlist.ParseFromIstream(&in_file);
        in_file.close();

        cv::Mat display(img.clone());
        for (const auto &bbox: bboxlist.bounding_boxes()) {
//            char ss[256];
//            sprintf(ss, "(%0.2f, %0.2f) - (%0.2f, %0.2f) @ %0.2f: %0.2f %s\n",
//                    bbox.top_left_x(), bbox.top_left_y(),
//                    bbox.bottom_right_x(), bbox.bottom_right_y(),
//                    bbox.azimuth(),
//                    bbox.scores(0), bbox.class_name().c_str());
//            std::cout << ss;
//            std::flush(std::cout);

            cv::rectangle(display,
                          cv::Point(bbox.top_left_x(), bbox.top_left_y()),
                          cv::Point(bbox.bottom_right_x(), bbox.bottom_right_y()),
                          cv::Scalar(255, 0, 0), 2);
            cv::putText(display,
                        bbox.class_name(),
                        cv::Point(bbox.top_left_x(), bbox.top_left_y() + 15),
                        CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255));
            cv::putText(display,
                        std::to_string(bbox.azimuth()),
                        cv::Point(bbox.top_left_x(), bbox.top_left_y() + 30),
                        CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 255));
        }
        cv::imshow("detection", display);

        cv::waitKey(10);
    }
}

