//
// Created by visionlab on 2/14/18.
//

#include "scene.h"

// stl
#include <iostream>
#include <fstream>
#include <io_utils.h>

// 3rd party
#include "glog/logging.h"
#include "opencv2/imgproc.hpp"
#include "tbb/parallel_for.h"
#include "folly/json.h"
#include "folly/dynamic.h"
#include "folly/FileUtil.h"

// own
#include "tracker_utils.h"

namespace feh {

namespace tracker {

void Scene::Update(const cv::Mat &evidence,
                   const vlslam_pb::BoundingBoxList &bbox_list,
                   const Sophus::SE3f &gwc,
                   const Sophus::SO3f &Rg,
                   const cv::Mat &img,
                   const std::string &imagepath) {
    ++frame_counter_;
    timer_.Tick("total");
    CHECK(initial_pose_set_) << "initial camera to world pose NOT set!!!";
    // STORE INPUTS
    gwc_ = gwc;
    Rg_ = Rg;
    img.copyTo(image_);
    evidence.copyTo(evidence_);
    input_bboxlist_.CopyFrom(bbox_list);

    // iterate over bounding boxes and check which bounding box is
    // not yet explained by the scene
    cv::Mat working_evidence(evidence.clone());
    for (const auto &category : valid_categories_) {
        vlslam_pb::BoundingBoxList category_bboxlist;
        float bbox_score_thresh = config_["bbox_score_thresh"][category].asDouble();
        float bbox_area_thresh = config_["bbox_area_thresh"][category].asDouble() * rows_ * cols_;
        for (const auto &bbox : bbox_list.bounding_boxes()) {
            if (bbox.class_name() == category
                && bbox.scores(0) > bbox_score_thresh
                && BBoxArea(bbox) > bbox_area_thresh
                && !BBoxTooCloseToBoundary(bbox)
                && !BBoxBadAspectRatio(bbox)) {
                // TODO: more strict tests
                vlslam_pb::BoundingBox *category_bbox = category_bboxlist.add_bounding_boxes();
                *category_bbox = bbox;
            }
        }
        // use existing objects to explain the evidence
        vlslam_pb::BoundingBoxList used_bboxlist;
        for (auto it = trackers_.begin(); it != trackers_.end();) {
            TrackerPtr tracker(*it);
            int used_bbox = tracker->Update(working_evidence,
                                            category_bboxlist,
                                            gwc_,
                                            Rg_,
                                            img,
                                            imagepath);

            if (used_bbox >= 0) {
                category_bboxlist.mutable_bounding_boxes(used_bbox)->set_label(-1); // SUPER HACK: RE-USE LABEL FIELD AS ACTIVE/INACTIVE FLAG
                vlslam_pb::BoundingBox *bbox_ptr = used_bboxlist.add_bounding_boxes();
                *bbox_ptr = category_bboxlist.bounding_boxes(used_bbox);
                // remove the portion of evidence which is explained by this tracker
//                auto rect = tracker->RectangleExplained();
//                std::cout << "tracker#" << tracker->id() << " explained bbox#" << used_bbox << "\n";
//                std::cout << "rect=" << rect << "\n";
//                working_evidence(rect).setTo(0);
                MaskOut<uint8_t>(working_evidence, tracker->RenderMask());
//                cv::imwrite("residual_after_tracker" + std::to_string(tracker->id()) + ".png", working_evidence);
                ++it;
            } else if (used_bbox == Tracker::kCompatibleBBoxNotFound) {
                if (tracker->status() == TrackerStatus::INITIALIZED) {
                    LOG(INFO) << "No compatible bbox found for"
                              << TermColor::bold + TermColor::green
                              << " initialized " << TermColor::end
                              << "tracker#" << tracker->id();
                    ++it;
                } else {
                    LOG(INFO) << "No compatible bbox found for tracker#"
                              << tracker->id();
                    ++it;
                }
            } else if (used_bbox == Tracker::kTooManyInitializationTrials) {
                LOG(INFO) << "too many" << TermColor::yellow << "initialization trials" << TermColor::end;
                it = trackers_.erase(it);

            } else if (used_bbox == Tracker::kTooManyNullObservations) {
                LOG(INFO) << "too many" << TermColor::red << "null observation" << TermColor::end;
                it = trackers_.erase(it);
            } else if (used_bbox == Tracker::kObjectOutOfView) {
                LOG(INFO) << "tracker#" << tracker->id()
                          << TermColor::cyan << " out of view" << TermColor::end;
                ++it;
            } else {
                LOG(FATAL) << "un-expected return value!!! of Tracker::Update";
            }

        }


        EliminateBadObjects();
        MergeObjects();

        ////////////////////////////////////////////////
        ////////////////////////////////////////////////
        ////////////////////////////////////////////////
        // remove false positive bounding boxes
        // which are not directly associated to an existing tracker, but
        // have large overlap with existing tracker(s)
        ////////////////////////////////////////////////
        ////////////////////////////////////////////////
        ////////////////////////////////////////////////
        // construct explained mask
        mask_.setTo(0);
        if (category_bboxlist.bounding_boxes_size() > 0) {
            for (TrackerPtr tracker : trackers_) {
                auto rect = tracker->RectangleExplained();
                cv::rectangle(mask_,
                              rect.tl(),
                              rect.br(),
                              1, -1);
            }
        }
        // iterate over bounding boxes
        for (int i = 0; i <  category_bboxlist.bounding_boxes_size(); ++i) {
            auto mutable_bbox = category_bboxlist.mutable_bounding_boxes(i);
            // if bbox i is explained by existing trackers
            if (mutable_bbox->label() == -1) {   // SUPER HACK: RE-USE LABEL FIELD AS ACTIVE/INACTIVE FLAG
                // already explained/used,
                // BUT NO EXPLICIT REMOVAL IS PERFORMED
                LOG(INFO) << TermColor::magenta << "bbox#" << i << " removed" << TermColor::end;
            } else {
                cv::Scalar covered_area = cv::sum(
                    mask_(cv::Rect(cv::Point((int) mutable_bbox->top_left_x(), (int) mutable_bbox->top_left_y()),
                                   cv::Point((int) mutable_bbox->bottom_right_x(), (int) mutable_bbox->bottom_right_y()))));
                double area = fabs((mutable_bbox->top_left_x() - mutable_bbox->bottom_right_x())
                                       * (mutable_bbox->top_left_y() - mutable_bbox->bottom_right_y()));
                if (covered_area[0] / area > 0.7) {
                    mutable_bbox->set_label(-1);
                    // NO EXPLICIT REMOVAL
                    LOG(INFO) << TermColor::magenta << "bbox#" << i << " removed" << TermColor::end;
                }
            }
        }

        ////////////////////////////////////////////////
        ////////////////////////////////////////////////
        ////////////////////////////////////////////////
        // for unexplained detections -- attempt to create new objects
        ////////////////////////////////////////////////
        ////////////////////////////////////////////////
        ////////////////////////////////////////////////
        for (const auto &bbox : category_bboxlist.bounding_boxes()) {
            if (bbox.label() != -1) {
                TrackerPtr new_tracker = std::make_shared<Tracker>();
                new_tracker->Initialize(config_["configure_files"][category].asString());
                new_tracker->build_visualization_ = false;
                new_tracker->SetInitCameraToWorld(gwc0_);
                new_tracker->InitializeFromBoundingBox(bbox, gwc_, Rg_);
                trackers_.push_back(new_tracker);
                LOG(INFO) << "new " << category << " created";
            }
        }
        // FIXME: SINCE WE HAVE ONLY ONE CATEGORY FOR NOW, ASSIGN CATEGORY_BBOXLIST TO INPUT_BBOXLIST
        input_bboxlist_.CopyFrom(category_bboxlist);
    }

    // LOG RESULTS
    UpdateLog();

    // UPDATE SEGMENTATION MASK
    UpdateSegMask();
//    UpdateSegMaskViz();

    ///////////////////////////////////////////////
    ///////////////////////////////////////////////
    ///////////////////////////////////////////////
    // REPORT
    ///////////////////////////////////////////////
    ///////////////////////////////////////////////
    ///////////////////////////////////////////////
    std::cout << "Total trackers = " << trackers_.size() << "\n";
    for (TrackerPtr tracker : trackers_) {
        std::cout << tracker->id() << " : "
                  << tracker->class_name() << " :status="
                  << as_integer(tracker->status());
        if (tracker->status() == TrackerStatus::OUT_OF_VIEW) {
            std::cout << tracker->mean_.transpose();
        }
        std::cout << "\n";

    }
    // VISUALIZATION
    Build2DView();
    timer_.Tock("total");
}

void Scene::UpdateSegMask() {
    zbuffer_.setTo(0);
    segmask_.setTo(-1);
    for (auto tracker : trackers_) {
        if (tracker->CentroidInCurrentView()(2) < 3)
        {
            cv::Mat depth = tracker->RenderDepth();
            auto op = [this, tracker, &depth](const tbb::blocked_range<int> &range) {
                for (int i = range.begin(); i < range.end(); ++i) {
                    for (int j = 0; j < depth.cols; ++j) {
                        float val(depth.at<float>(i, j));
                        if (val > 0) {
                            // only on foreground
                            float zbuf_val(this->zbuffer_.at<float>(i, j));
                            if (zbuf_val == 0 || val < zbuf_val) {
                                this->zbuffer_.at<float>(i, j) = val;
                                this->segmask_.at<int32_t>(i, j) = tracker->id();
                            }
                        }
                    }
                }};
            tbb::parallel_for(tbb::blocked_range<int>(0, depth.rows), op);
        }
    }

    // Update visibility information for each tracker
    for (auto tracker : trackers_) {
//        if (tracker->status() != TrackerStatus::OUT_OF_VIEW)
        {
            tracker->ResetVisibility();
            tracker->UpdateVisibility(segmask_);
        }
    }
}

//void Scene::UpdateSegMaskViz() {
//    zbuffer_viz_.setTo(0);
//    segmask_viz_.setTo(-1);
//    for (auto tracker : trackers_) {
////        if (tracker->status() == TrackerStatus::INITIALIZED)
//        {
//            cv::Mat depth = tracker->RenderDepth();
//            auto op = [this, tracker, &depth](const tbb::blocked_range<int> &range) {
//                for (int i = range.begin(); i < range.end(); ++i) {
//                    for (int j = 0; j < depth.cols; ++j) {
//                        float val(depth.at<float>(i, j));
//                        if (val > 0) {
//                            // only on foreground
//                            float zbuf_val(this->zbuffer_viz_.at<float>(i, j));
//                            if (zbuf_val == 0 || val < zbuf_val) {
//                                this->zbuffer_viz_.at<float>(i, j) = val;
//                                this->segmask_viz_.at<int32_t>(i, j) = tracker->id();
//                            }
//                        }
//                    }
//                }};
//            tbb::parallel_for(tbb::blocked_range<int>(0, depth.rows), op);
//        }
//    }
//}

void Scene::UpdateLog() {
    folly::dynamic obj_array = folly::dynamic::array;
    for (auto tracker : trackers_) {
        folly::dynamic tracker_obj = folly::dynamic::object;
        tracker_obj["id"] = tracker->id();
        tracker_obj["model_name"] = tracker->shape_name();
        tracker_obj["status"] = as_integer(tracker->status());
        io::WriteMatrixToDynamic(tracker_obj, "model_pose", tracker->pose().block<3, 4>(0, 0));
        obj_array.push_back(tracker_obj);
    }
    folly::dynamic data = folly::dynamic::object(std::to_string(frame_counter_), obj_array);
//    if (!obj_array.empty())
    {
        log_.push_back(obj_array);
    }
}
void Scene::WriteLogToFile(const std::string &filename) {
    folly::writeFile(folly::toPrettyJson(log_), filename.c_str());

}

bool Scene::BBoxTooCloseToBoundary(const vlslam_pb::BoundingBox &bbox) const {
    bool bad1 = bbox.top_left_x() > 0.9 * cols_ || bbox.top_left_y() > 0.9 * rows_
        || bbox.bottom_right_x() < 0.1 * cols_ || bbox.bottom_right_y() < 0.1 * rows_;

    bool bad2 = bad1;

    bad2 = bad2 || bbox.top_left_x() < 0.01 * cols_ || bbox.top_left_y() < 0.01 * rows_
    || bbox.bottom_right_x() > 0.99 * cols_ || bbox.bottom_right_y() > 0.99 * rows_;

    return bad2;
}

bool Scene::BBoxBadAspectRatio(const vlslam_pb::BoundingBox &bbox) const {
    float size_x = fabs(bbox.top_left_x() - bbox.bottom_right_x());
    float size_y = fabs(bbox.top_left_y() - bbox.bottom_right_y());
    float ratio = size_x / (size_y + 1e-4);
    return ratio > 4 || ratio < 0.25;
}

void Scene::MergeObjects() {
    for (auto it = trackers_.begin(); it != trackers_.end(); ++it) {
        for (auto it2 = std::next(it); it2 != trackers_.end(); ) {
            auto c1 = (*it)->pose().block<3, 1>(0, 3);
            auto c2 = (*it2)->pose().block<3, 1>(0, 3);
            if ((c1.head<2>()-c2.head<2>()).norm() < 0.5
                && as_integer((*it)->status()) >= as_integer((*it2)->status())) {
                // IF IT2 HAS HIGHER PRIORITY, DO NOT REMOVE IT
                it2 = trackers_.erase(it2);
                std::cout << TermColor::bold+TermColor::red << "MERGING REDUDANT OBJECT" << TermColor::endl;
                std::cout << "c1=" << c1.transpose() << "c2=" << c2.transpose() << "\n";

            } else ++it2;
        }
    }
}

void Scene::EliminateBadObjects() {
    for (auto it = trackers_.begin(); it != trackers_.end(); ) {
        Vec3f c = (*it)->CentroidInCurrentView();
        if ((*it)->status() != TrackerStatus::OUT_OF_VIEW && c(2) <= 0.0 ) {
            std::cout << TermColor::bold+TermColor::red << "ELIMINATE OBJECTS CLOSE TO THE CAMERA: #" << (*it)->id() << TermColor::endl;
            it = trackers_.erase(it);
        } else {
            cv::Mat mask = (*it)->RenderMask();
            int size = mask.rows * mask.cols;
            auto total = std::count_if(mask.data, mask.data + size,
                                       [](uint8_t p) { return p == 0;});
            if (total > 0.5 * size) {
                std::cout << TermColor::bold+TermColor::red << "ELIMINATE OBJECTS COVERING MOST OF THE IMAGE: #" << (*it)->id() << TermColor::endl;
                it = trackers_.erase(it);
            } else ++it;
        }
    }
}

} // NAMESPACE TRACKER
}   // NAMESPACE FEH

