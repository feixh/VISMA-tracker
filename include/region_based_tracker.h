//
// Created by visionlab on 11/2/17.
//

#include "eigen_alias.h"
#include "utils.h"

// stl
#include <string>
#include <vector>

// 3rd party
#include "opencv2/opencv.hpp"
#include "folly/dynamic.h"
#include "sophus/se3.hpp"

//own
#include "renderer.h"
#include "distance_transform.h"

namespace feh {
namespace tracker {

class RegionBasedTracker {
public:
    RegionBasedTracker();
    void Initialize(const std::string &config_file,
                    const std::vector<float> &camera_params = std::vector<float>{},
                    const std::vector<float> &vertices = std::vector<float>{},
                    const std::vector<int> &faces = std::vector<int>{});
    /// \brief: Optimize object pose given the image, a rough bounding box
    /// and an initial guess.
    /// \param image: Image to work with.
    /// \param bbox: a bounding box roughly covers the object.
    /// \param gm_init: initial guess of model to camera rigid body transformation.
    /// \return: Optimized pose.
    Sophus::SE3f Optimize(const cv::Mat &image,
                          const cv::Rect &bbox,
                          const Sophus::SE3f &gm_init);
    /// \brief: Optimize object pose given image and a rough bounding box.
    /// Initial guess will be read from configure file.
    /// \return: Optimized pose.
    Sophus::SE3f Optimize(const cv::Mat &image,
                          const cv::Rect &bbox);


    /// \brief: Initialize object pose.
    /// \param image: Input image.
    /// \param bbox: Bounding box.
    /// \param gm_init: Initial pose of the model.
    void InitializeTracker(const cv::Mat &image,
                    const cv::Rect &bbox,
                    const Sophus::SE3f &gm_init);

    /// \brief: Update pose estimation for tracking.
    void Update(const cv::Mat &image);

    bool UpdateOneStepAtLevel(int level, Sophus::SE3f &g);

    /// \brief: Compute color histograms from a given bounding box.
    /// Pixels inside the bounding box are considered as foreground.
    /// Pixels on the expaned band of the box are background pixels.
    void ComputeColorHistograms(const cv::Mat &image,
                                const cv::Rect &bbox,
                                cv::Rect &inflated_bbox,
                                std::vector<VecXf> &histf,
                                std::vector<VecXf> &histb) const;
    /// \brief: Compute color histograms from a given mask.
    /// Pixels inside the mask are considered as foreground.
    /// Pixels in the rest of an expanded box are background.
    void ComputeColorHistograms(const cv::Mat &image,
                                const cv::Mat &mask,
                                cv::Rect &bbox,
                                std::vector<VecXf> &histf,
                                std::vector<VecXf> &histb) const;

    cv::Mat GetDisplay() { return display_; }

private:
    folly::dynamic config_;
    Timer timer_;
    std::vector<float> vertices_;
    std::vector<int> faces_;
    std::vector<RendererPtr> renderers_;
    DistanceTransform distance_transformer_;
    std::vector<cv::Mat> image_pyr_;    // image pyramid
    std::vector<cv::Mat> depth_;    // depth maps
    std::vector<cv::Mat> mask_; // projection masks
    std::vector<cv::Mat> contour_;  // object contours
    std::vector<cv::Mat> distance_; // distance fields
    // x (1st slice) and y (2nd slice) coordinates of closest edge pixel
    std::vector<cv::Mat> distance_index_;
    std::vector<cv::Mat> signed_distance_;  // signed distance field
    // d(signed_distance_field) / d(xp)
    std::vector<cv::Mat> dsdf_dxp_;
    // heaviside field (1st slice) and
    // its derivative w.r.t. signed distance field (2nd slice)
    std::vector<cv::Mat> heaviside_;
    std::vector<cv::Rect> roi_; // region of interest at different levels
    // posterior of a pixel belonging to foreground (1st slice)
    // and background (2nd slice)
    std::vector<std::vector<cv::Mat>> P_;
    float fx_, fy_, cx_, cy_;
    int rows_, cols_;
    int levels_;
    int inflate_size_;  // to what extent expand the mask
    int histogram_size_;
    std::vector<VecXf> hist_f_;   // foreground color histograms
    std::vector<VecXf> hist_b_;   // background color histograms
    float alpha_f_, alpha_b_;   // foreground/background histogram learning rate
    std::vector<std::string> hist_code_;    // color code of the histograms
    std::unordered_map<uint64_t, Eigen::Matrix<float, 2, 6>> dxp_dtwist_;

    bool constrain_rotation_;
    cv::Mat display_;

    Sophus::SE3f gm_;

};


}   // namespace tracker

}   // namespace feh
