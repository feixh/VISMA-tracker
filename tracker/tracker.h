//
// Created by visionlab on 10/17/17.
//
#pragma once
#include "alias.h"

// stl
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "math.h"

// sophus
#include "json/json.h"
#include "opencv2/highgui/highgui.hpp"

// lcm
#include "lcm/lcm-cpp.hpp"

// own
#include "renderer.h"
#include "vlslam.pb.h"
#include "oned_search.h"
#include "distance_transform.h"
#include "particle.h"
#include "lcm_msg_handlers.h"
#include "se3.h"

namespace feh {
namespace tracker {

enum class TrackerStatus : int {
    INVALID = 0,
    VALID,
    INITIALIZING,
    INITIALIZED,
    OUT_OF_VIEW,
    CONVERGED,
    FINALIZED,
    BAD
};

struct Shape {
    std::string name_;  // name of the CAD model
    // Some objects are deformable, we only use the dominant rigid part for inference.
    // For instance, the upper part of a swivel chair can rotate which is usually the visible part of a chair,
    // we only use the upper part of the chair for inference.
    MatXf vertices_, part_vertices_;
    MatXi faces_, part_faces_;
    std::vector<RendererPtr> render_engines_;
};
using ShapeId = int;

class Tracker {
public:
    friend class Scene;
    Tracker();
    ~Tracker();
    /// \brief: Initialize tracker from a given jason configuration.
    void Initialize(const std::string &config_file,
                    const Json::Value &more_config=Json::Value{});
    /// \brief: Initialize object from a given bounding box.
    /// \param bbox: From which we initialize object pose.
    /// \param cam_pse: g(init camera <- current camera).
    /// \param Rg: Rotation to align gravity to (0, 0, 1)
    /// \param imagepath: Input image full path,
    /// which is only passed around as part of bbox message.
    void InitializeFromBoundingBox(const vlslam_pb::BoundingBox &bbox,
                                   const SE3 &cam_pose,
                                   const SO3 &Rg,
                                   std::string imagepath="");
    /// \brief: Initialization with azimuth distribution returned by CNN as prior.
    void InitializeFromBoundingBoxWithAzimuthPrior(const vlslam_pb::BoundingBox &bbox,
                                                   const SE3 &cam_pose,
                                                   const SO3 &Rg,
                                                   std::string imagepath="");

    bool IsObjectPoseInitialized() const { return status_ >= TrackerStatus::INITIALIZING; }

    /// \brief: Given evidence, update state of the tracker.
    /// \param evidence: Edge map returned by CNN.
    /// \param bbox_list: List of bounding boxes returned by object detector.
    /// \param current_to_init: Transformation from current camera frame to initial camera frame.
    // FIXME: replace int with enum type to reflect status
    int Update(const cv::Mat &evidence,
               const vlslam_pb::BoundingBoxList &bbox_list,
               const SE3 &gwc,
               const SO3 &Rg,
               const cv::Mat &img,
               std::string imagepath="");
    /// \brief: Given hypothesis (v) and pixelwise posterior map (P), compute
    /// log likelihood.
    float ComputeAppearanceLikelihood(const Vec4f &v, const std::vector<cv::Mat> &P);
    /// \brief: Scale, crop evidences, etc.
    int Preprocess(const cv::Mat &evidence,
                   const vlslam_pb::BoundingBoxList &bbox_list,
                   const SE3 &gwc,
                   const SO3 &Rg,
                   const cv::Mat &img);
    /// \brief: Compute the normal of edge pixels at given level.
    void ComputeEdgeNormal(int level);
    /// \brief: Compute the normal of edge pixels at all levels.
    void ComputeEdgeNormalAllLevel();
    /// \brief: Compute the quality of the current state, including but not limited to matching ratio,
    /// mean matching distance, CNN score at the expectation.
    void ComputeQualityMeasure();
    struct {
        float matching_ratio_;
        float mean_matching_distance_;
        float CNN_score_;
        int since_last_label_change_;
        Vec4f uncertainty_;
    } quality_;
    /// \brief: Check convergence of estimators based on history.
    bool StationaryEnough();
    bool CloseEnough(float ratio_thresh, float dist_thresh, float score_thresh=0.5);
    /// \brief: Check whether the object is out-of-view?
    bool IsOutOfView();
    /// \brief: Compute log-likelihood and related info (match ratio & average match distance).
    /// \param info: (match_ratio, average_match_distance).
    /// \param match_ratio_order: usually order 1.
    float LogLikelihoodFromEdgelist(const std::vector<EdgePixel> &edgelist,
                                    std::array<float, 2> *info=nullptr,
                                    float match_ratio_order=1);
//    /// \brief: Update visibility properties, also dependent on other objects in the scene.
//    /// \param visible_ratio: Ratio of visible area over total projection area.
//    void UpdateVisibility(float visible_ratio);
//    /// \brief: Update the visible region.
//    /// \param x, y: pixel coordinates of a visible pixel.
//    void UpdateVisibility(int x, int y);
    /// \brief: Update visibility information by providing the instance segmentation mask.
    /// The mask is constructed by projecting shapes at their respective optimal pose.
    void UpdateVisibility(const cv::Mat &segmask);
    /// \brief: Reset visibility parameters.
    void ResetVisibility();
    /// \brief: Rectangular region explained by the tracker.
    cv::Rect RectangleExplained(int level=0);
    void SwitchMeshForInference();
    void SwitchMeshForVisualization();

    /// \brief: Render edge map at current best estimate.
    cv::Mat Render(int level=0);
    cv::Mat RenderAt(const SE3 &object_pose, int level=0);
    cv::Mat RenderWireframe(int level=0);
    cv::Mat RenderWireframeAt(const SE3 &object_pose, int level=0);
    cv::Mat RenderDepth(int level=0);
    cv::Mat RenderDepthAt(const SE3 &object_pose, int level=0);
    cv::Mat RenderMask(int level=0);
    cv::Mat RenderMaskAt(const SE3 &object_pose, int level=0);
    Vec2f Project(const Vec3f &vertex, int level=0) const;
    void GetProjection(std::vector<Vec2f> &projections, int level=0) const;
    Vec2f ProjectMean(int level=0) const;
    Vec3f CentroidInCurrentView() {
        Vec3f v(mean_.head<3>());
        v(2) = std::exp(v(2));
        v.head<2>() *= v(2);
        v = gwc_.inv() * gwr_ * v;
        return v;
    }

    ////////////////////////////////////////////////
    /// HANDY ACCESSORS
    ////////////////////////////////////////////////
    uint32_t id() const { return id_; }
    const std::string &class_name() const { return class_name_; }
    const std::string &shape_name() const { return shapes_.at(best_shape_match_).name_; }
    Mat4f pose() {
        gwm_ = gwr_ * SE3(MatForRender());
        return gwm_.matrix();
    }
    TrackerStatus status() const { return status_; }
    float visible_ratio() const { return visible_ratio_; }
    int matched_bbox() const { return best_bbox_index_; }
    float max_iou() const { return max_iou_; }
    const MatXf &vertices(int i=-1) const { return shapes_.at(i == -1 ? best_shape_match_ : i).vertices_; }
    const MatXi &faces(int i=-1) const { return shapes_.at(i < 0 ? best_shape_match_ : i).faces_; }
    float fx(int lvl=0) const { return fx_[lvl]; }
    float fy(int lvl=0) const { return fy_[lvl]; }
    float cx(int lvl=0) const { return cx_[lvl]; }
    float cy(int lvl=0) const { return cy_[lvl]; }
    int rows(int lvl=0) const { return rows_[lvl]; }
    int cols(int lvl=0) const { return cols_[lvl]; }
    /// \brief: Return the minimum enclosing rectangle of visible part of the object.
    cv::Rect visible_region() const { return cv::Rect(cv::Point(visible_tl_(0), visible_tl_(1)),
                                                      cv::Point(visible_br_(0), visible_br_(1))); }

    const cv::Mat &GetFilterView() const { return display_; }
    void WriteOutParticles(const std::string &filename) const;

    void SetInitCameraToWorld(const SE3 &gwc0) {
        gwc0_ = gwc0;
    }
    /// \brief: Convert from local parametrization to object pose whose
    /// translational part is defined in reference camera frame and azimuth
    /// is defined in inertial frame.
    Mat4f MatForRender(const Vec4f &v) const;
    Mat4f MatForRender() const;

private:
    void ScaleMat(cv::Mat &mat) const;
    void BuildFilterView();
    void LogDebugInfo();


    /// \brief: particle filter update step; call the following function
    /// 1. ComputePropsoals: propose new particles
    /// 2. ComputeLikelihood:
    /// 3. ComputePrior:
    void PFUpdate(int level=-1);
    void MultiScalePFUpdate();
    int ComputeProposals(int level=-1);
    void ComputeLikelihood(int level=-1);
    void ComputePrior(int level=-1);
    // publish bounding box proposals to be evaluated in network process via LCM port
    void PublishBBoxProposals(const std::vector<cv::Rect> &rect_list);
    /// \brief: Make Monte Carlo move on azimuth estimation to explore symmetry of objects.
    void MakeMonteCarloMove(int level=-1);

    void EKFUpdate();
    void EKFInitialize();

    // FIXME: make it private
private:
    // system state
    const uint32_t id_;
    int initialization_counter_;
    int no_observation_counter_;
    int convergence_counter_;
    int hits_;
    TrackerStatus status_, saved_status_;
    float ts_, last_update_ts_;
    bool use_partial_mesh_;

    // camera model & render engine
    Json::Value config_;

    std::shared_ptr<lcm::LCM> port_;
    std::shared_ptr<BBoxLikelihoodHandler> handler_;

//    std::shared_ptr<UndistorterPTAM> undistorter_;
    //FIXME: ideally render engines are wrapped into Shape class, need to eliminate the following
    // stack of render engines conditioned on most probable shape id
    std::vector<RendererPtr> renderers_;
//    RendererPtr renderer_; // renderer for downsampled size
//    RendererPtr renderer0_; // renderer for original size
    std::shared_ptr<std::knuth_b> generator_;
    Timer timer_;
    OneDimSearch oned_search_;
    DistanceTransform distance_transform_;
    std::string class_name_;

    std::vector<ShapeId> shape_ids_;    // set of possible shape ids
    std::unordered_map<ShapeId, Shape> shapes_; // set of corresponding shapes
    int best_shape_match_;

    // camera related parameters
    std::vector<int> rows_, cols_;
    std::vector<float> fx_, fy_, cx_, cy_;
    float s_;

    // multi-scale parameters
    int scale_level_;
    float scale_factor_;    // target scaling factor
    int evidence_kernel_size_, prediction_kernel_size_;

    // likelihood parameters
    bool use_CNN_, use_MC_move_;
    std::vector<float> log_likelihood_weight_, log_prior_weight_, log_proposal_weight_, CNN_log_likelihood_weight_;
    float CNN_prob_thresh_;
    float keep_id_prob_;    // probability of keeping the current shape id
    float azi_flip_rate_;   // flip rate of azimuth in MC move
    float azi_uniform_mix_;

    // bounding boxes
    vlslam_pb::BoundingBoxList bbox_list0_;
    cv::Rect best_bbox_;

    // particle filter parameters
    Vec4f mean_, init_state_;
    std::vector<Vec4f> history_;
    std::vector<ShapeId> label_history_;
    Vec4f initial_std_;   // std of the Gaussian centered at the initial guess
    Vec4f proposal_std_;  // std of the transition/proposal distribution
    VecXf azimuth_prob_;    // azimuth distribution returned by CNN
    int azimuth_offset_;

    // extended kalman filter parameters
    Mat4f P_;   // state covariance

    // particles and weights
    int max_num_particles_;
    int total_visible_edgepixels_;
    Particles<float, 4> particles_;

    // visibility properties
    float visible_ratio_;
    Vec2i visible_tl_, visible_br_;
    cv::Mat visible_mask_;
    int best_bbox_index_;
    float max_iou_;

    // current image address
    std::string image_fullpath_;
    // reference camera pose
    SE3 gwc0_; // initial camera frame to world frame
    SE3 gwr_;     // g(world <- reference camera)
    // current camera pose g(reference <- current)
    SE3 gwc_;     // g(world <- current camera)
    SE3 grc_;      // g(reference camera <- current camera)
    SO3 Rg_;
    SE3 gwm_;  // g(world <- model) For output

    std::ofstream dbg_file_;

public:
    // FIXME: handy options -- finally should be private
    bool build_visualization_;

private:
    // buffers
    cv::Mat display_;

    std::vector<cv::Mat> image_, evidence_, evidence_dir_, edge_buffer_;
public:
    // constants
    static const int kCompatibleBBoxNotFound;
    static const int kTooManyInitializationTrials;
    static const int kTooManyNullObservations;
    static const int kObjectOutOfView;
    static uint32_t tracker_counter_;
};

typedef std::shared_ptr<Tracker> TrackerPtr;

}   // namespace tracker
}   // namespace feh


