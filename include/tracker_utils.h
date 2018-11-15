//
// Created by feixh on 10/18/17.
//
#pragma once
#include "alias.h"

// stl
#include <chrono>
#include <unordered_map>
#include <memory>

// 3rd party
#include "opencv2/core/core.hpp"

// own
#include "utils.h"
#include "vlslam.pb.h"
#include "oned_search.h"

namespace feh {

namespace tracker {
/// \brief: Convert from local parametrization (inverse depth) to usual state representation.
Vec4f StateFromLocalParam(const Vec4f &local_param, Mat4f *jac=nullptr);
///// \brief: Convert from state to local parametrization (inverse depth).
//Vec4f LocalParamFromState(const Vec4f &state);
/// \brief: Convert from state to homogeneous pose matrix.
Mat4f Mat4FromState(const Vec4f &state);
/// \brief: Make the vertices centered at origin.
void NormalizeVertices(MatXf &V);
void ScaleVertices(MatXf &V, float scale_factor);
/// \brief: Rotate the vertices such that positive x of canonical object frame is consistent with positive x of identity camera frame.
/// This is needed for the pre-scanned models.
void RotateVertices(MatXf &V, float angle);
/// \brief: Flip y-axis of vertices, needed by ShapeNet models.
void FlipVertices(MatXf &V);
/// \brief: Overlay the predicted single-channel edge map onto RGB image.
/// \param mask: the mask, 0 means background, > 0 means foreground
/// \param image: to which render the overlayed image
/// \param invert_mask: if true, 0 means foreground, > 0 means background
/// \param color: color of the mask
void OverlayMaskOnImage(const cv::Mat &mask, cv::Mat &image, bool invert_mask=false, const uint8_t *color=nullptr);
/// \brief: Make it easier to visualize depth map by removing invalid depth values and linearize valid ones.
void PrettyDepth(cv::Mat &out, float z_near = 0.05f, float z_far = 20.0f);
/// \brief: Convert from radian to an integer number which is azimuth index.
int AzimuthIndexFromRadian(float rad);
/// \brief: From azimuth index to radian.
float RadianFromAzimuthIndex(int index);
/// \brief: Warp angle to [0, 2 \pi)
float WarpAngle(float angle);
/// \brief: Compute the area covered by the bounding box.
float BBoxArea(const vlslam_pb::BoundingBox &bbox);
/// \brief: Compute the smallest bounding box which covers the contour/mask.
cv::Rect RectEnclosedByContour(const std::vector<EdgePixel> &edgelist, int rows, int cols);
float ComputeIoU(cv::Rect r1, cv::Rect r2);
cv::Rect InflateRect(const cv::Rect &rect, int rows=480, int cols=640, int pad=8);

const uint8_t kColorGreen[] = {0, 255, 0};
const uint8_t kColorRed[] = {0, 0, 255};
const uint8_t kColorBlue[] = {255, 0, 0};
const uint8_t kColorCyan[] = {255, 255, 0};
const uint8_t kColorYellow[] = {0, 255, 255};
const uint8_t kColorMagenta[] = {255, 0, 255};
const uint8_t kColorWhite[] = {255, 255, 255};
const std::unordered_map<std::string, const uint8_t*> kColorMap = {
    {"chair", kColorGreen},
    {"couch", kColorGreen},
    {"car", kColorGreen},
    {"truck", kColorGreen}
};

/// \brief: Compute foreground and background per channel color distribution.
/// The distributions are non-parametric, represented by 3 histograms (RGB channels).
/// \param image: Input image.
/// \param bbox: Bounding box of the foreground object.
/// \param inflated_bbox: Return value, inflated object bounding box.
/// \param histf: Return value, color histogram of foreground pixels.
/// \param histb: Return value, color histogram of background pixels.
/// \param histogram_size: Number of bins of each histogram.
/// \param inflate_size: Number of pixels in each direction to expand the input bounding box,
/// to include more background pixels.
void ComputeColorHistograms(const cv::Mat &image,
                            const cv::Rect &bbox,
                            cv::Rect &inflated_bbox,
                            std::vector <VecXf> &histf,
                            std::vector <VecXf> &histb,
                            int histogram_size = 32,
                            int inflate_size = 8);

/// \brief: Compute foreground and background per channel color distribution.
/// The distributions are non-parametric, represented by 3 histograms (RGB channels).
/// \param image: Input RGB image.
/// \param mask: Binary object mask, foreground is 0, background is >0.
/// \param inflated_bbox: Return value, inflated object bounding box.
/// \param histf: Return value, color histogram of foreground pixels.
/// \param histb: Return value, color histogram of background pixels.
/// \param histogram_size: Number of bins of each histogram.
/// \param inflate_size: Number of pixels in each direction to expand the input bounding box,
/// to include more background pixels.
void ComputeColorHistograms(const cv::Mat &image,
                            const cv::Mat &mask,
                            cv::Rect &bbox,
                            std::vector<VecXf> &histf,
                            std::vector<VecXf> &histb,
                            int histogram_size = 32,
                            int inflate_size = 8);

/// \brief: Given an image and foreground & background color distribution (by histograms), compute
/// the posterior of each pixel (in a given region) being on foreground and background.
/// \param image: Input RGB image.
/// \param hist_f: foreground color histogram.
/// \param hist_b: background color histogram.
/// \param bbox: Bounding box indicating the region to apply this computation.
void ComputePixelwisePosterior(const cv::Mat &image,
                               std::vector<cv::Mat> &P,
                               std::vector<VecXf> &hist_f,
                               std::vector<VecXf> &hist_b,
                               const cv::Rect &bbox);
/// \brief: Given an image and foreground & background color distribution (by histograms), compute
/// the posterior of each pixel (in the whole image) being on foreground and background.
/// \param image: Input RGB image.
/// \param hist_f: foreground color histogram.
/// \param hist_b: background color histogram.
void ComputePixelwisePosterior(const cv::Mat &image,
                               std::vector<cv::Mat> &P,
                               std::vector<VecXf> &hist_f,
                               std::vector<VecXf> &hist_b);

/// \brief: Make float type zbuffer to char type
cv::Mat PrettyZBuffer(const cv::Mat &zbuffer, float z_near=0.05f, float z_far=20.0f);
/// \brief: Linearize a depth map for visualization.
cv::Mat LinearizeDepthMap(const cv::Mat &zbuffer, float z_near=0.05f, float z_far=20.0f);
cv::Mat PrettyLabelMap(const cv::Mat &label_map, const std::vector<std::array<uint8_t, 3>> &color_map);

}   // namespace tracker

}   // namespace feh
