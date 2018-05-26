//
// Created by feixh on 10/31/17.
//
#pragma once
#include "common/eigen_alias.h"

// stl
#include <vector>
#include <list>
#include <string>
#include <unordered_set>

// 3rd party
#include "sophus/se3.hpp"
#include "opencv2/core.hpp"

// own
#include "common/utils.h"
#include "tracker.h"
#include "vlslam.pb.h"

namespace feh {

namespace tracker {

class Scene {
public:
    Scene();
    void Initialize(const std::string &config_file,
                    const folly::dynamic &more_config=folly::dynamic{});
    void SetInitCameraToWorld(const Sophus::SE3f &gwc0);

    /// \brief: Entrance of semantic mapping.
    /// \param evidence: Edge map extracted using modified SegNet.
    /// \param bbox_list: List of object proposals given by Faster R-CNN.
    /// \param gwc: Camera to world transformation.
    /// \param Rg: Rotation of gravity.
    /// \param img: Input RGB image.
    /// \param imagepath: Full path of the input image to inform CNN process which image to operate on.
    void Update(const cv::Mat &evidence,
                const vlslam_pb::BoundingBoxList &bbox_list,
                const Sophus::SE3f &gwc,
                const Sophus::SO3f &Rg,
                const cv::Mat &img,
                const std::string &imagepath);

    /// \brief: Update result log.
    void UpdateLog();
    void WriteLogToFile(const std::string &filename);

    /// \brief: Update the segmentation mask by constructing z-buffer, etc.
    void UpdateSegMask();
    /// \brief: Update segmentation for visualization.
    void UpdateSegMaskViz();
    /// \brief: Merge objects close in 3D.
    void MergeObjects();
    /// \brief: Eliminate those too close to current camera frame.
    void EliminateBadObjects();

    void Build2DView();
    const cv::Mat &Get2DView() const { return display_; };
    const cv::Mat &GetZBuffer() const { return zbuffer_; }
    const cv::Mat &GetSegMask() const { return segmask_; }
    bool BBoxTooCloseToBoundary(const vlslam_pb::BoundingBox &bbox) const;
    bool BBoxBadAspectRatio(const vlslam_pb::BoundingBox &bbox) const;

private:
    int frame_counter_;
    bool initial_pose_set_;
    std::unordered_set<std::string> valid_categories_;
    std::list<TrackerPtr> trackers_;
    folly::dynamic config_;
    folly::dynamic log_;
    Timer timer_;


    int rows_, cols_;
    Sophus::SE3f gwc0_, gwc_;
    Sophus::SO3f Rg_;

    // BUFFER
    cv::Mat mask_;  // binary explanation mask
    cv::Mat zbuffer_, zbuffer_viz_;   // global z-buffer for occlusion reasoning
    cv::Mat segmask_, segmask_viz_; // pixel-wise instance label assignment
    cv::Mat display_;   // for display -- edge thumbnail & contour/mask imposed on input image
    cv::Mat image_; // input image
    cv::Mat evidence_;  // input evidence
    vlslam_pb::BoundingBoxList input_bboxlist_;
    static std::vector<std::array<uint8_t, 3>> random_color_;
};


template <typename T>
void MaskOut(cv::Mat &input, const cv::Mat &mask) {
    CHECK_EQ(input.rows, mask.rows);
    CHECK_EQ(input.cols, mask.cols);
    for (int i = 0; i < mask.rows; ++i)
        for (int j = 0; j < mask.cols; ++j)
            if (mask.at<uint8_t>(i, j) <= 0)
                input.at<T>(i, j) = 0;
}



}

}
