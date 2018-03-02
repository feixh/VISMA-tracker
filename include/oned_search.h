//
// Created by feixh on 10/24/17.
//
#pragma once
// stl
#include <vector>
#include <iostream>

// 3rd party
#include "opencv2/core.hpp"
#include "tbb/blocked_range.h"

// own
#include "common/utils.h"


namespace feh {

struct EdgePixel {
    float x, y, dir, depth;
    static const int DIM = 4;
};

static EdgePixel InvalidEdgePixel() {
    return {-1, -1, -1, -1};
}
struct OneDimSearchMatch {
    OneDimSearchMatch():
        dist_(-1),
        edgepixel_(InvalidEdgePixel()){}

    OneDimSearchMatch(float x1, float y1,
                 float x2, float y2,
                 float affinity=-1):
        pt1_(x1, y1),
        pt2_(x2, y2),
        dist_(sqrt((x1-x2) * (x1-x2) + (y1-y2) * (y1-y2))),
        affinity_(affinity),
        edgepixel_(InvalidEdgePixel())
    {}

    OneDimSearchMatch(const Vec2f &pt1, const Vec2f &pt2,
                 float affinity=-1):
        pt1_(pt1),
        pt2_(pt2),
        dist_((pt1-pt2).norm()),
        affinity_(affinity)
    {}

    void Set(const Vec2f &pt1, const Vec2f &pt2, float affinity=-1) {
        pt1_ = pt1;
        pt2_ = pt2;
        dist_ = (pt1 - pt2).norm();
        affinity_ = affinity;
    }

    void Set(float x1, float y1, float x2, float y2, float affinity=-1) {
        Set(Vec2f(x1, y1), Vec2f(x2, y2), affinity);
    }

    void Set(const EdgePixel &edgepixel) { edgepixel_ = edgepixel; }

    bool IsValid() const { return dist_ >= 0; }
    bool HasEdgePixel() const {
        return edgepixel_.x == -1 && edgepixel_.y == -1
            && edgepixel_.dir == -1 && edgepixel_.depth == -1;
    }

    Vec2f pt1_, pt2_;
    float dist_;
    float affinity_; // extra measures
    EdgePixel edgepixel_;   // reference edge pixel
};

/// \brief: 1-dimensional search along normal of edge pixels.
class OneDimSearch {
public:
    OneDimSearch():
        step_length_(1),
        search_line_length_(50),
        intensity_threshold_(200),
        direction_consistency_thresh_(0),
        parallel_(false)
    {}

    /// \brief: Given a list of edge pixels and an edgemap, find matches.
    /// \param edgelist: List of edge pixels from the rendered edge map.
    /// \param target: Target edge map to match.
    /// \param matches: Output matches.
    /// \param target_dir: Gradient direction of the target edge map.
    void operator()(const std::vector<EdgePixel> &edgelist,
                    const cv::Mat &target,
                    std::vector<OneDimSearchMatch> &matches,
                    const cv::InputArray &target_dir = std::vector<int>{}) const;

    /// \brief: Parallel version of above.
    void Parallel(const std::vector<EdgePixel> &edgelist,
                  const cv::Mat &target,
                  std::vector<OneDimSearchMatch> &matches,
                  const cv::InputArray &target_dir = std::vector<int>{}) const;
    // options
    int step_length_;
    int search_line_length_;
    uint8_t intensity_threshold_;
    float direction_consistency_thresh_;
    bool parallel_;

public:

    /// \brief: Build visualization of matching, also shows matches one by one.
    static
    void BuildMatchView(const cv::Mat &ref_img,
                        const cv::Mat &target_img,
                        const std::vector<OneDimSearchMatch> &matches,
                        cv::Mat &out);

    /// \brief: Visualize normals with colors (like optical flow visualization).
    static
    void BuildNormalView(const std::vector<EdgePixel> &edgelist,
                         cv::Mat &normal_view);

    /// \brief: Visualize normals with short line segments.
    static
    void BuildNormalView2(const std::vector<EdgePixel> &edgelist,
                          cv::Mat &normal_view);

};

}   // feh

