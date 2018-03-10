//
// Created by feixh on 10/31/17.
//

#include "scene.h"

// 3rd party
#include "opencv2/imgproc.hpp"
#include "folly/json.h"
#include "folly/FileUtil.h"

// own
#include "tracker_utils.h"

namespace feh {

namespace tracker {

std::vector<std::array<uint8_t, 3>> Scene::random_color_ = GenerateRandomColorMap<8>();

Scene::Scene() :
    frame_counter_(0),
    initial_pose_set_(false),
    timer_("Scene") {
    // can only handle chair now
    valid_categories_.insert("chair");
//    valid_categories_.insert("couch");
//    valid_categories_.insert("table");
    timer_.report_average = true;
}


void Scene::Initialize(const std::string &config_file) {
    std::string content;
    folly::readFile(config_file.c_str(), content);
    config_ = folly::parseJson(folly::json::stripComments(content));
    // LOAD SCENE CONFIGURATION
    log_ = folly::dynamic::array();

    // LOAD CAMERA CONFIGURATION
    folly::readFile(config_["camera_config"].asString().c_str(), content);
    config_["camera"] = folly::parseJson(folly::json::stripComments(content));

    auto cam_cfg = config_["camera"];
//    // FIXME: MAYBE WRAP THE FOLLOWING INTO A CAMERA INSTANCE
//    float fx_          = cam_cfg["fx"].asDouble();
//    float fy_          = cam_cfg["fy"].asDouble();
//    float cx_          = cam_cfg["cx"].asDouble();
//    float cy_          = cam_cfg["cy"].asDouble();
//    float s_           = cam_cfg["s"].asDouble();
//    float z_near_      = cam_cfg["z_near"].asDouble();
//    float z_far_      = cam_cfg["z_far"].asDouble();
    rows_        = cam_cfg["rows"].asInt();
    cols_        = cam_cfg["cols"].asInt();

    // ALLOCATE BUFFERS
    mask_ = cv::Mat(rows_, cols_, CV_8UC1);
    zbuffer_ = cv::Mat(rows_, cols_, CV_32FC1);
    zbuffer_viz_ = cv::Mat(rows_, cols_, CV_32FC1);
    segmask_ = cv::Mat(rows_, cols_, CV_32SC1);
    segmask_viz_ = cv::Mat(rows_, cols_, CV_32SC1);
    image_ = cv::Mat(rows_, cols_, CV_8UC3);
    evidence_ = cv::Mat(rows_, cols_, CV_8UC1);
}

void Scene::SetInitCameraToWorld(const Sophus::SE3f &gwc0) {
    if (initial_pose_set_) {
        LOG(FATAL) << "initial camera pose already set!!!";
    }
    gwc0_ = gwc0;
    initial_pose_set_ = true;
}



void Scene::Build2DView() {
    display_ = image_.clone();
    // list all the trackers
    for (TrackerPtr tracker : trackers_) {
        if (tracker->status() == TrackerStatus::OUT_OF_VIEW) continue;
//        if (config_["visualization"]["show_mean_boundary"].asBool()) {
//            cv::Mat boundary = tracker->RenderAtCurrentEstimate(tracker->renderer0_ptr());
//            OverlayMaskOnImage(boundary, display_, false);
//        }

        if (config_["visualization"]["show_mean_boundary"].asBool()) {
            auto prediction = tracker->Render(0);
            auto rc = random_color_[tracker->id()+1];
            OverlayMaskOnImage(prediction, display_, false, &rc[0]); //kColorMap.at(tracker->class_name()));
        }

        if (config_["visualization"]["show_projections"].asBool()) {
            std::vector<Vec2f> projections;
            tracker->GetProjection(projections, 0);
            for (const auto &pt : projections) {
                cv::circle(display_, cv::Point(pt(0), pt(1)), 1, cv::Scalar(0, 0, 255), -1);
            }
        }

        if (config_["visualization"]["show_mean_projection"].asBool()) {
            Vec2f mean_proj = tracker->ProjectMean(0);
            const uint8_t *cp = kColorMap.at(tracker->class_name());
            cv::Scalar color(cp[0], cp[1], cp[2]);
            cv::circle(display_,
                       cv::Point(mean_proj(0), mean_proj(1)),
                       4, color, -1);
            cv::putText(display_, std::to_string(tracker->id()),
                        cv::Point(mean_proj(0), mean_proj(1)-10),
                        CV_FONT_HERSHEY_PLAIN, 1,
                        color);
        }

        if (config_["visualization"]["show_visibility_box"].asBool()) {
            auto rc = random_color_[tracker->id()+1];
            cv::rectangle(display_, tracker->visible_region(), cv::Scalar(rc[0], rc[1], rc[2]), 2);
        }
    }


    // AUGMENT PANEL SHOWS INPUT EDGEMAP, ZBUFFER, INSTANCE MASK AND TRACKER STATUS
    cv::Mat augment_display(display_.rows >> 2, display_.cols, CV_8UC3);
    augment_display.setTo(0);

    if (config_["visualization"]["show_bounding_boxes"].asBool()) {
        for (int i = 0; i < input_bboxlist_.bounding_boxes_size(); ++i) {
            auto bbox = input_bboxlist_.bounding_boxes(i);
            if (valid_categories_.count(bbox.class_name())) {
                const uint8_t *cp = kColorMap.at(bbox.class_name());
//                cv::Scalar color(random_color_[bbox_idx][0],
//                                 random_color_[bbox_idx][1],
//                                 random_color_[bbox_idx][2]);
                cv::Scalar color(cp[0], cp[1], cp[2], 50);
                cv::rectangle(display_,
                              cv::Point(bbox.top_left_x(), bbox.top_left_y()),
                              cv::Point(bbox.bottom_right_x(), bbox.bottom_right_y()),
                              color, 2);
                cv::putText(display_,
                            bbox.class_name(),
                            cv::Point(bbox.bottom_right_x()-20, bbox.bottom_right_y() - 30),
                            CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255));
                cv::putText(display_,
                            std::to_string(i),
                            cv::Point(bbox.bottom_right_x()-20, bbox.bottom_right_y() - 15),
                            CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 255));
            }
        }
    }

    int tbn_counter(0);
    int tbn_rows(display_.rows >> 2), tbn_cols(display_.cols >> 2);
    // INPUT EDGE MAP
    if (config_["visualization"]["show_evidence_as_thumbnail"].asBool()) {
        cv::Mat thumbnail;
        cv::resize(evidence_, thumbnail, cv::Size(tbn_cols, tbn_rows));
        cv::cvtColor(thumbnail, thumbnail, CV_GRAY2RGB);
        cv::Rect rect(tbn_counter * tbn_cols, 0,
                      tbn_cols, tbn_rows);
        thumbnail.copyTo(augment_display(rect));
        cv::rectangle(augment_display, rect, cv::Scalar(255, 255, 0), 2);
        ++tbn_counter;
    }
    // ZBUFFER
    if (config_["visualization"]["show_zbuffer_as_thumbnail"].asBool()) {
        cv::Mat thumbnail;
        cv::resize(PrettyZBuffer(zbuffer_), thumbnail, cv::Size(tbn_cols, tbn_rows));
        cv::cvtColor(thumbnail, thumbnail, CV_GRAY2RGB);
        cv::Rect rect(tbn_counter * tbn_cols, 0,
                      tbn_cols, tbn_rows);
        thumbnail.copyTo(augment_display(rect));
        cv::rectangle(augment_display, rect, cv::Scalar(255, 255, 0), 2);
        ++tbn_counter;
    }
    // INSTANCE MASK
    if (config_["visualization"]["show_segmask_as_thumbnail"].asBool()) {
        cv::Mat thumbnail;
        cv::resize(PrettyLabelMap(segmask_, random_color_), thumbnail, cv::Size(tbn_cols, tbn_rows));
        cv::Rect rect(tbn_counter * tbn_cols, 0,
                      tbn_cols, tbn_rows);
        thumbnail.copyTo(augment_display(rect));
        cv::rectangle(augment_display, rect, cv::Scalar(255, 255, 0), 2);
    }

    // TRACKER STATUS
    cv::Point2i debug_info_pos(display_.cols-(display_.cols >> 2), 10);
    for (TrackerPtr tracker : trackers_) {
        std::string status_str = folly::sformat("T#{}-B#{}<{:0.2f}> s:{} v:{:0.2f} ({:0.1f}, {:0.1f})",
                                                tracker->id(),
                                                tracker->matched_bbox(),
                                                tracker->max_iou(),
                                                as_integer(tracker->status()),
                                                tracker->visible_ratio(),
                                                tracker->ProjectMean(0)[0],
                                                tracker->ProjectMean(0)[1]);

        auto rc = random_color_[tracker->id() + 1];
        cv::Scalar color(rc[0], rc[1], rc[2]);
        cv::putText(augment_display, status_str, debug_info_pos, CV_FONT_HERSHEY_PLAIN, 0.5, color, 1);
        debug_info_pos.y += 12;
    }

    // BOTTOM PANEL SHOWS FPS AND OTHER SYSTEM STATS
    cv::Mat bottom_panel(12, display_.cols, CV_8UC3);
    bottom_panel.setTo(0);
    std::string sys_stats = folly::sformat("#OBJ:{}  FPS:{:0.2f}",
                                           trackers_.size(),
                                           1000.0f / (timer_.LookUp("total", Timer::MILLISEC, true)+eps));
    cv::putText(bottom_panel, sys_stats, cv::Point(0, 10), CV_FONT_HERSHEY_PLAIN, 1, {0, 255, 0});

    // COMPOSE PANELS
    cv::Mat full_display(augment_display.rows + display_.rows + bottom_panel.rows, display_.cols, CV_8UC3);
    augment_display.copyTo(full_display(cv::Rect(0, 0, augment_display.cols, augment_display.rows)));
    display_.copyTo(full_display(cv::Rect(0, augment_display.rows, display_.cols, display_.rows)));
    bottom_panel.copyTo(full_display(cv::Rect(0, augment_display.rows+display_.rows, bottom_panel.cols, bottom_panel.rows)));
    full_display.copyTo(display_);
}


}   // namespace tracker
}   // namespace feh

