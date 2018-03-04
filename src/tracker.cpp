//
// Created by visionlab on 10/17/17.
//
#include <tracker.h>
#include "tracker.h"

// 3rd party
#include "opencv2/imgproc.hpp"
#include "folly/FileUtil.h"
#include "folly/json.h"
#include "folly/Format.h"

// own
#include "io_utils.h"
#include "tracker_utils.h"
#include "parallel_kernels.h"

namespace feh {
namespace tracker {

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// STATUS CONSTANT
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
const int Tracker::kCompatibleBBoxNotFound = -1;
const int Tracker::kTooManyInitializationTrials = -2;
const int Tracker::kTooManyNullObservations = -3;
const int Tracker::kObjectOutOfView = -4;

uint32_t Tracker::tracker_counter_ = 0;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// CONSTRUCTOR/DESTRUCTOR PAIR
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

Tracker::Tracker() :
    id_(tracker_counter_++),
    initialization_counter_(0),
    no_observation_counter_(0),
    convergence_counter_(0),
    hits_(0),
    status_(TrackerStatus::INVALID),
    saved_status_(TrackerStatus::INVALID),
    ts_(0),
    use_partial_mesh_(false),
    port_(nullptr),
    generator_(nullptr),
    timer_("tracker"),
    class_name_(""),
    scale_level_(0),
    scale_factor_(1.0),
    evidence_kernel_size_(0),
    prediction_kernel_size_(0),
    use_CNN_(false),
    use_MC_move_(false),
    CNN_prob_thresh_(0.0),
    max_num_particles_(500),
    total_visible_edgepixels_(0),
    visible_ratio_(1.0f),
    visible_tl_(10000, 10000),
    visible_br_(0, 0),
    build_visualization_(true) {

}

Tracker::~Tracker() {
    if (dbg_file_.is_open()) {
        dbg_file_.close();
    }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// INITIALIZER USING JSON CONFIGURE FILES
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void Tracker::Initialize(const std::string &config_file) {
    if (status_ == TrackerStatus::VALID) {
        LOG(FATAL) << "FATAL::each tracker can only be initialized ONCE!!!";
    }

    // root of json
    std::string content;
    folly::readFile(config_file.c_str(), content);
    config_ = folly::parseJson(folly::json::stripComments(content));



    // setup hyper-parameters of filter
    auto filter_cfg = config_["filter"];
    max_num_particles_ = filter_cfg["initialization_num_particles"].asInt();
    keep_id_prob_      = filter_cfg["keep_shape_id_probability"].asDouble();
    azi_flip_rate_     = filter_cfg["azimuth_flip_rate"].asDouble();  // flip azimuth with this probability
    azi_uniform_mix_   = filter_cfg["azimuth_uniform_mix"].asDouble();
    initial_std_       = io::GetVectorFromDynamic<float, 4>(filter_cfg, "initial_std");
    proposal_std_      = io::GetVectorFromDynamic<float, 4>(filter_cfg, "proposal_std");
    azimuth_prob_.setZero(360);
    LOG(INFO) << "max num of particles=" << max_num_particles_;
    LOG(INFO) << "initial std=" << initial_std_.transpose();
    LOG(INFO) << "proposal std=" << proposal_std_.transpose();

    // setup likelihood parameters
    use_partial_mesh_          = filter_cfg["use_partial_mesh"].asBool();
    use_CNN_                   = filter_cfg["use_CNN"].getBool();
    use_MC_move_               = filter_cfg["use_MC_move"].getBool();
    CNN_prob_thresh_           = filter_cfg["CNN_probability_threshold"].asDouble();
    evidence_kernel_size_      = filter_cfg["evidence_blur_kernel_size"].asInt();
    prediction_kernel_size_    = filter_cfg["prediction_blur_kernel_size"].asInt();
    scale_level_               = filter_cfg["scale_level"].asInt();
    log_likelihood_weight_.resize(scale_level_, filter_cfg["log_likelihood_weight"].asDouble());
    CNN_log_likelihood_weight_.resize(scale_level_,  filter_cfg["CNN_log_likelihood_weight"].asDouble());
    log_prior_weight_.resize(scale_level_, filter_cfg["log_prior_weight"].asDouble());
    log_proposal_weight_.resize(scale_level_, filter_cfg["log_proposal_weight"].asDouble());
    for (int i = 1; i < scale_level_; ++i) {
        double factor = pow(0.5, i);
        log_likelihood_weight_[i] *= factor;
        CNN_log_likelihood_weight_[i] *= factor;
//        log_prior_weight_[i] *= factor;
//        log_proposal_weight_[i] *= factor;
    }
    if (use_CNN_) LOG(INFO) << TermColor::green << "CNN likelihood is ON" << TermColor::endl;
    CHECK_GT(CNN_prob_thresh_, 0);


    // setup one-dimensional search
    auto oned_cfg = config_["oned_search"];
    oned_search_.step_length_                  = oned_cfg["step_length"].asInt();
    oned_search_.search_line_length_           = oned_cfg["search_line_length"].asInt();
    oned_search_.intensity_threshold_          = uint8_t(oned_cfg["intensity_thresh"].asInt() & 0xff);
    oned_search_.direction_consistency_thresh_ = oned_cfg["direction_thresh"].asDouble();
    oned_search_.parallel_                     = oned_cfg["parallel"].getBool();


    // camera parameters

//    in_file.open(config_["camera_config"].asString(), std::ios::in);
//    CHECK(in_file.is_open());
//    in_file >> config_["camera"];
//    in_file.close();
    folly::readFile(config_["camera_config"].asString().c_str(), content);

    config_["camera"] = folly::parseJson(folly::json::stripComments(content));

    auto cam_cfg = config_["camera"];
    s_           = cam_cfg["s"].asDouble();
    for (int i = 0; i < scale_level_; ++i) {
        fx_.push_back(i == 0? cam_cfg["fx"].asDouble() : fx_.back()*0.5f);
        fy_.push_back(i == 0? cam_cfg["fy"].asDouble() : fy_.back()*0.5f);
        cx_.push_back(i == 0? cam_cfg["cx"].asDouble() : cx_.back()*0.5f);
        cy_.push_back(i == 0? cam_cfg["cy"].asDouble() : cy_.back()*0.5f);
        rows_.push_back(i == 0? cam_cfg["rows"].asInt() : (rows_.back() >> 1));
        cols_.push_back(i == 0? cam_cfg["cols"].asInt() : (cols_.back() >> 1));
    }

    // setup shapes
    // FIXME: for now just us integers as shape ids
    auto cad_list = io::LoadMeshDatabase(config_["CAD_database_root"].asString(), config_["CAD_category_json"].asString());
    for (int i = 0; i < cad_list.size(); ++i) {
        shape_ids_.push_back(i);
        shapes_[i].name_ = cad_list[i].substr(0, cad_list[i].find('.'));

        try {
            std::string mesh_file = config_["CAD_database_root"].asString() + "/" + cad_list[i] + ".obj";
            std::tie(shapes_[i].vertices_, shapes_[i].faces_) = io::LoadMeshFromObjFile(mesh_file);
        } catch (std::exception &e) {
            std::cout << TermColor::red << e.what() << TermColor::endl;
        }

        if (use_partial_mesh_) {
            try {
                std::string mesh_file = config_["CAD_database_root"].asString() + "/" + cad_list[i] + "_part.obj";
                std::tie(shapes_[i].part_vertices_, shapes_[i].part_faces_) = io::LoadMeshFromObjFile(mesh_file);
            } catch (std::exception &e){
                std::cout << TermColor::red << e.what() << TermColor::endl;
            }
        }
        // render engine not set yet
    }
    best_shape_match_ = config_["hack"].getDefault("best_shape_match", 0).asInt();

    // near and far plane
    float z_near, z_far;
    z_near = cam_cfg["z_near"].asDouble();
    z_far = cam_cfg["z_far"].asDouble();

    // scaling factor of the target level relative to input image
    scale_factor_ = powf(0.5, scale_level_-1);
    // setup a bank of renderers
    for (int i = 0; i < scale_level_; ++i) {
        int search_line_len = oned_cfg["search_line_length"].asInt();
        for (int sid : shape_ids_) {
            RendererPtr new_renderer = std::make_shared<Renderer>(rows_[i], cols_[i]);
            new_renderer->SetCamera(z_near, z_far, fx_[i], fy_[i], cx_[i], cy_[i]);
            new_renderer->SetOneDimSearch(search_line_len,
                                          oned_cfg["intensity_thresh"].asInt(),
                                          oned_cfg["direction_thresh"].asDouble());
            if (!shapes_.at(sid).part_vertices_.empty()) {
                new_renderer->SetMesh(shapes_.at(sid).part_vertices_, shapes_.at(sid).part_faces_);
            } else {
                new_renderer->SetMesh(shapes_.at(sid).vertices_, shapes_.at(sid).faces_);
            }
            // render engine setup here
            shapes_.at(sid).render_engines_.push_back(new_renderer);
        }
        // DO NOT TOUCH!!! THE FOLLOWING SETUP PERFORMS REASONABLY WELL
        search_line_len /= 1.414;
    }
    renderers_ = shapes_.at(best_shape_match_).render_engines_;

    // setup multi-level evidence buffer
    for (int i = 0; i < scale_level_; ++i) {
        evidence_.push_back(cv::Mat(rows_[i], cols_[i], CV_8UC1));
        evidence_dir_.push_back(cv::Mat(rows_[i], cols_[i], CV_32FC1));
        image_.push_back(cv::Mat(rows_[i], cols_[i], CV_8UC3));
    }
    visible_mask_ = cv::Mat(rows_[0], cols_[0], CV_8UC1);
    visible_mask_.setTo(1);

    // setup random number generator
    if (config_["fixed_seed"].getBool()) {
        generator_ = std::make_shared<std::knuth_b>(0);
    } else {
        generator_ = std::make_shared<std::knuth_b>(time(NULL));
    }
    particles_.Initialize(generator_);


    // set flag
    status_ = TrackerStatus::VALID;

    // setup port for inter process communication
    if (use_CNN_) {
        handler_ = std::make_shared<BBoxLikelihoodHandler>();
        port_    = std::make_shared<lcm::LCM>();
        port_->subscribe("likelihood", &BBoxLikelihoodHandler::Handle, handler_.get());
        if (port_->good()) {
            LOG(INFO) << TermColor::green << "LCM port ready to go" << TermColor::end;
        } else {
            LOG(FATAL) << TermColor::red << "failed to setup LCM port" << TermColor::end;
        }
    }

    // setup debug file
    if (config_["debug_info"].getDefault("save_to_file", false).getBool())
    {
        std::string dbg_filename = folly::sformat("dbg_tracker{:04d}.txt", id());
        dbg_file_.open(dbg_filename, std::ios::out);
        CHECK(dbg_file_.is_open());
    }
}



////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// CORE TRACKER UTILITIES
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
int Tracker::Update(const cv::Mat &in_evidence,
                    const vlslam_pb::BoundingBoxList &bbox_list,
                    const Sophus::SE3f &gwc,
                    const Sophus::SO3f &Rg,
                    const cv::Mat &img,
                    std::string imagepath) {
    ts_ += 1;
    image_fullpath_ = imagepath;


    timer_.Tick("preprocess");
    int used_bbox_index = Preprocess(in_evidence, bbox_list, gwc, Rg, img);
    timer_.Tock("preprocess");

    if (status_ == TrackerStatus::OUT_OF_VIEW && !IsOutOfView()) {
        status_ = saved_status_;
        saved_status_ = TrackerStatus::INVALID;
    }

#ifdef FEH_MULTI_OBJECT_MODEL
    if (used_bbox_index == kCompatibleBBoxNotFound) {
        if (visible_ratio_ > 0.8
            || (visible_ratio_ < 0.1 && convergence_counter_ <= 0)) ++no_observation_counter_;
        if (no_observation_counter_
            > config_.getDefault("max_num_allowed_null_observations", 10).asInt()
                + 20 * convergence_counter_ + 10 * hits_) {
            used_bbox_index = kTooManyNullObservations;
//                used_bbox_indexurn kTooManyNullObservations;
        } else {
            used_bbox_index = kCompatibleBBoxNotFound;
//                return kCompatibleBBoxNotFound;
        }
    } else {
        no_observation_counter_ = 0;
    }

     if (status_ != TrackerStatus::OUT_OF_VIEW) {
        if (status_ == TrackerStatus::INITIALIZED
            || status_ == TrackerStatus::INITIALIZING) {

            if (status_ == TrackerStatus::INITIALIZING
                && visible_ratio_ > 0.8f) {
                ++initialization_counter_;
                if (initialization_counter_
                    > 5 * hits_ + config_.getDefault("max_num_allowed_initialization", 150).asInt()) {
                    return kTooManyInitializationTrials;
                }
                timer_.Tick("particle update");
                MultiScalePFUpdate();
                ComputeQualityMeasure();
                timer_.Tock("particle update");

                // ESTABLISH CONVERGENCE CRITERION HERE
                if (StationaryEnough() && CloseEnough(0.85, 10, 0.0)) {
                    status_ = TrackerStatus::INITIALIZED;
                    proposal_std_ = io::GetVectorFromDynamic<float, 4>(config_["filter"], "small_proposal_std");
                     azi_uniform_mix_ = 0;
                    initial_std_ *= 10;
                    keep_id_prob_ = 1-1e-4;
                    init_state_ = mean_;
                    use_CNN_ = true;
                    use_MC_move_ = false;
                    azi_flip_rate_ = 0.01;
                    particles_.Subsample(config_["filter"]["tracking_num_particles"].asInt());
                    timer_.Reset();
                    initialization_counter_ = 0;

//                    for (auto kv : shapes_) {
//                        for (auto r : kv.second.render_engines_) {
//                            r->SetOneDimSearch(20, 64, 0.6);
//                        }
//                    }
                }

            } else if (status_ == TrackerStatus::INITIALIZED
                && visible_ratio_ > 0.8f
                && used_bbox_index >=0
                &&  ts_ < last_update_ts_ + 50 ) {
                ++convergence_counter_;
                LOG(INFO) << "running in small proposal distribution mode\n";
                // TURN OFF UPDATE STEP AFTER CONVERGENCE
                // timer_.Tick("particle update");
                 MultiScalePFUpdate();
                ComputeQualityMeasure();
                last_update_ts_ = ts_;
                // timer_.Tock("particle update");

                if (status_ == TrackerStatus::OUT_OF_VIEW) {
                    no_observation_counter_ = 0;
                    return kObjectOutOfView;
                } else {
                    // FIXME: need to tune parameters
                    if (!CloseEnough(0.75, 15, 0.0)) {
                        status_ = TrackerStatus::INITIALIZING;
                        auto filter_cfg = config_["filter"];
                        keep_id_prob_ = filter_cfg["keep_shape_id_probability"].asDouble();
                        initial_std_ = io::GetVectorFromDynamic<float, 4>(filter_cfg, "initial_std");
                        proposal_std_ = io::GetVectorFromDynamic<float, 4>(filter_cfg, "proposal_std");
                        scale_level_ = filter_cfg["scale_level"].asInt();

                        float keep_prob = filter_cfg["keep_shape_id_probability"].asDouble();
                        float chance_over_time = std::count_if(label_history_.begin(), label_history_.end(),
                                                               [this](const ShapeId &id) {
                                                                   return id == this->best_shape_match_;
                                                               })
                            / (label_history_.size() + eps);
//                        keep_id_prob_ = keep_prob + 0.9f * (1 - keep_prob) * chance_over_time;
                        init_state_ = mean_;
                        azi_uniform_mix_ = filter_cfg["azimuth_uniform_mix"].asDouble();



//                    use_MC_move_ = true;
                        use_CNN_ = true;
//                    azi_flip_rate_     = filter_cfg["azimuth_flip_rate"].asDouble();
                        // FIXME: PROPER RE-INITIALIZATION
                        particles_.resize(max_num_particles_ * shape_ids_.size(), {mean_, best_shape_match_, 0.0f});

                        timer_.Reset();
                        initialization_counter_ = 0;

//                        for (auto kv : shapes_) {
//                            for (auto r : kv.second.render_engines_) {
//                                r->SetOneDimSearch(config_["oned_search"].getDefault("search_line_length", 40).getInt(),
//                                                   config_["oned_search"].getDefault("intensity_thresh", 64).getInt(),
//                                                   config_["oned_search"].getDefault("direction_thresh", 0.85).getDouble());
//                            }
//                        }
                    }
                }
            }

        } else {
            LOG(INFO) << TermColor::yellow << "Out-Of-View" << TermColor::endl;
        }
#else
        PFUpdate();
#endif

    }

    if (status_ != TrackerStatus::OUT_OF_VIEW && IsOutOfView()) {
//        if (convergence_counter_ > 0)
        {
            saved_status_ = status_;
            status_ = TrackerStatus::OUT_OF_VIEW;
//        no_observation_counter_ = 0;
            if (used_bbox_index >= 0) used_bbox_index = kObjectOutOfView;
        }
//        else used_bbox_index = kTooManyInitializationTrials;
    }


    // switch to full meshes for visualization
    SwitchMeshForVisualization();

    // build visualization
    if (build_visualization_) {
        timer_.Tick("visualize");
        BuildFilterView();
        timer_.Tock("visualize");
    }

    DLOG(INFO) << "mean=" << mean_.transpose() << "\n";
    if (config_["debug_info"]["print_timing"].getBool()) {
        std::cout << timer_;
    }

    return used_bbox_index;
}


void Tracker::MultiScalePFUpdate() {
    auto stored_proposal_std = proposal_std_;
    proposal_std_ *= powf(1.04, scale_level_-1);
    auto saved_search_line_length = oned_search_.search_line_length_;

    for (int level = scale_level_-1; level >= 0; --level) {
        oned_search_.search_line_length_ >>= 1;
        for (auto kv : shapes_) {
            for (auto r : kv.second.render_engines_) {
                r->SetOneDimSearch(oned_search_.search_line_length_);
            }
        }
        PFUpdate(level);
        proposal_std_ /= 1.04;
    }
    proposal_std_ = stored_proposal_std;
    oned_search_.search_line_length_ = saved_search_line_length;
};

int Tracker::Preprocess(const cv::Mat &in_evidence,
                        const vlslam_pb::BoundingBoxList &bbox_list,
                        const Sophus::SE3f &gwc,
                        const Sophus::SO3f &Rg,
                        const cv::Mat &img) {
    // use partial meshes during inference
    SwitchMeshForInference();

    // set current camera pose
    gwc_ = gwc;
    grc_ = gwr_.inverse() * gwc;
    for (auto sid : shape_ids_) {
        for (auto r: shapes_.at(sid).render_engines_) {
            r->SetCamera(grc_.inverse().matrix());
        }
    }

    // Find region proposals of which the class label is consistent with the estimated class label.
    bbox_list0_.CopyFrom(bbox_list);

    cv::Rect rect = visible_region();
    if (rect.area() == 0) {
        std::vector<EdgePixel> edgelist;
        renderers_[0]->ComputeEdgePixels(MatForRender(mean_), edgelist);
        rect = RectEnclosedByContour(edgelist, rows_[0], cols_[0]);
    }

    timer_.Tick("compute iou");
    // FIXME: better to use IoU instead of counting in-box edge pixels
    // Find the candidate bounding box which has most overlap with the projection.
    float max_iou(0);
    int best_bbox_index(kCompatibleBBoxNotFound);
    int counter(0);
    for (const auto &bbox : bbox_list0_.bounding_boxes()) {
        CHECK_EQ(bbox.class_name(), class_name_ );
        if (bbox.label() != -1) {   // SUPER HACK: RE-USE LABEL FIELD AS ACTIVE/INACTIVE FLAG
            // scale bounding boxes accordingly
            float iou = ComputeIoU(rect,
                                   cv::Rect(cv::Point((int)bbox.top_left_x(),
                                                      (int)bbox.top_left_y()),
                                            cv::Point((int)bbox.bottom_right_x(),
                                                      (int)bbox.bottom_right_y())));

            if ( iou > max_iou ) {
                max_iou = iou;
                best_bbox_index = counter;
            }
        }
        ++counter;
    }
    timer_.Tock("compute iou");

    if (best_bbox_index != kCompatibleBBoxNotFound) {
        // Not enough overlap, set not found.
        if (max_iou < 0.5
            || (status_ == TrackerStatus::INITIALIZED
            && max_iou  < 0.7)) {
            best_bbox_index = kCompatibleBBoxNotFound;
        } else {
            // FIXME: Restrict evidence to the most probable bounding box.
            auto tmp = bbox_list0_.bounding_boxes(best_bbox_index);
            best_bbox_ = cv::Rect(cv::Point(tmp.top_left_x(), tmp.top_left_y()),
                                  cv::Point(tmp.bottom_right_x(), tmp.bottom_right_y()));
        }
    }


    timer_.Tick("prepare evidence");
    ////////////////////////////////////////
    // SETUP EVIDENCE AND EDGE NORMALS
    ////////////////////////////////////////
    image_[0]    = img.clone();
    // PREDICTION GUIDED EVIDENCE
//    if (status_ == TrackerStatus::INITIALIZED) {
//        for (int i = 0; i < evidence_[0].rows; ++i)
//            for (int j = 0; j < evidence_[0].cols; ++j)
//                evidence_[0].at<uint8_t>(i, j) = visible_mask_.at<uint8_t>(i, j) > 0 ? in_evidence.at<uint8_t>(i, j) : 0;
//    } else {
//        if (best_bbox_index >= 0) {
//            evidence_[0].setTo(0);
//            in_evidence(best_bbox_).copyTo(evidence_[0](best_bbox_));
//        } else {
//            in_evidence.copyTo(evidence_[0]);
//        }
////        in_evidence.copyTo(evidence_[0]);
//    }

    in_evidence.copyTo(evidence_[0]);

//    cv::imwrite(folly::sformat("evidence_{:04d}_{:04d}.png", id_, (int)ts_), evidence_[0]);

    // scale image and evidence
    for (int i = 1; i < scale_level_; ++i) {
        cv::Size sz(image_[i].cols, image_[i].rows);
        cv::pyrDown(image_[i-1], image_[i], sz);
        cv::pyrDown(evidence_[i-1], evidence_[i], sz);
    }
    ComputeEdgeNormalAllLevel();
    for (auto sid : shape_ids_) {
        std::vector<RendererPtr> render_engines{shapes_.at(sid).render_engines_};
        for (int lvl = 0; lvl < render_engines.size(); ++lvl) {
            render_engines[lvl]->UploadEvidence(evidence_[lvl].data);
            render_engines[lvl]->UploadEvidenceDirection((float*)evidence_dir_[lvl].data);
        }
    }
    ////////////////////////////////////////
    timer_.Tock("prepare evidence");



    best_bbox_index_ = best_bbox_index;
    max_iou_         = max_iou;

    if (best_bbox_index >= 0) ++hits_;

    return best_bbox_index;
}

void Tracker::ComputeEdgeNormalAllLevel() {
    for (int level = 0; level < scale_level_; ++level) {
        ComputeEdgeNormal(level);
    }
}

void Tracker::ComputeEdgeNormal(int level) {
    timer_.Tick("evidence gradient");
    cv::Mat dx, dy;
    cv::Sobel(evidence_[level], dx, CV_32F, 1, 0, 3);
    cv::Sobel(evidence_[level], dy, CV_32F, 0, 1, 3);
    tbb::parallel_for(tbb::blocked_range<int>(0, evidence_[level].rows),
                      [this, level, &dx, &dy](const tbb::blocked_range<int> &range) {
                          for (int i = range.begin(); i < range.end(); ++i) {
                              for (int j = 0; j < evidence_dir_[level].cols; ++j) {
                                  float direction = atan2(dy.at<float>(i, j), dx.at<float>(i, j));
                                  if (!std::isnan(direction)) {
                                      evidence_dir_[level].at<float>(i, j) = direction;
                                  }
                              }
                          }
                      },
                      tbb::auto_partitioner());
    timer_.Tock("evidence gradient");
}


bool Tracker::IsOutOfView() {
    // select renderer -- workong on the coarsest level
    int level = scale_level_-1;
    auto renderer = renderers_[level];
    Vec2f center = ProjectMean(level);
    if (center(0) < renderer->cols() * 0.05
            || center(1) < renderer->rows() * 0.05
            || center(0) > renderer->cols() * 0.95
            || center(1) > renderer->rows() * 0.95
        && convergence_counter_ > 0) {
        return true;
    }

    std::vector<EdgePixel> edgelist;
    renderer->ComputeEdgePixels(MatForRender(mean_), edgelist);
    cv::Rect rect = RectEnclosedByContour(edgelist, rows_[level], cols_[level]);

    if (rect.tl().x > cols_[level] * 0.90f
        || rect.tl().y > rows_[level] * 0.90
        || rect.br().x < cols_[level] * 0.10
        || rect.br().y < rows_[level] * 0.10) {
        return true;
    }

    auto c = CentroidInCurrentView();
    if (c(2) > 2.5 && status_ != TrackerStatus::INITIALIZING) return true;

    return false;
}


cv::Rect Tracker::RectangleExplained(int level) {
    auto renderer = renderers_[level];
    std::vector<EdgePixel> edgelist;
    renderer->ComputeEdgePixels(MatForRender(mean_), edgelist);
    return RectEnclosedByContour(edgelist, rows_[level], cols_[level]);
}

void Tracker::UpdateVisibility(const cv::Mat &segmask) {
    cv::Mat mask = RenderMask();
    visible_mask_.setTo(0);
    int size = visible_mask_.rows * visible_mask_.cols;
    auto total = std::count_if(mask.data, mask.data + size,
                               [](uint8_t p) { return p == 0;});
    int visible = 0;
    for (int i = 0; i < segmask.rows; ++i) {
        for (int j = 0; j  <segmask.cols; ++j) {
            if (segmask.at<int32_t>(i, j) == id_) {
                ++visible;
                if (j < visible_tl_(0)) visible_tl_(0) = j;
                if (j > visible_br_(0)) visible_br_(0) = j;
                if (i < visible_tl_(1)) visible_tl_(1) = i;
                if (i > visible_br_(1)) visible_br_(1) = i;
                visible_mask_.at<uint8_t>(i, j) = 255;
            }
        }
    }
    visible_ratio_ = visible / (total + eps);
    cv::GaussianBlur(visible_mask_, visible_mask_, cv::Size(5, 5), 0);

    // DEBUG:
//    cv::imwrite(folly::sformat("visible_mask_{:04d}_{:04d}.png", id_, (int)ts_), visible_mask_);
}

void Tracker::ResetVisibility() {
    visible_ratio_ = 1.0f;
    visible_tl_ << 10000, 100000;
    visible_br_ << 0, 0;
    visible_mask_.setTo(0);
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// I/O FOR VISUALIZATION AND DEBUGGING
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
void Tracker::BuildFilterView() {
//    RendererPtr renderer;
//    if (level > 0) {
//        renderer = shapes_.at(best_shape_match_).render_engines_.back();
//        image_.back().copyTo(display_);
//    } else {
//        renderer = shapes_.at(best_shape_match_).render_engines_.front();
//        image_.front().copyTo(display_);
//    }
    int level = 0;
    image_[level].copyTo(display_);

    vlslam_pb::BoundingBoxList *the_bbox_list;
    the_bbox_list = &bbox_list0_;

    if (config_["visualization"]["show_bounding_boxes"].getBool()
        && status_ == TrackerStatus::INITIALIZING) {
        for (const auto &bbox: the_bbox_list->bounding_boxes()) {
            cv::rectangle(display_,
                          cv::Point(bbox.top_left_x(), bbox.top_left_y()),
                          cv::Point(bbox.bottom_right_x(), bbox.bottom_right_y()),
                          cv::Scalar(255, 0, 0), 2);
            cv::putText(display_,
                        bbox.class_name(),
                        cv::Point(bbox.top_left_x(), bbox.top_left_y() + 15),
                        CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 0, 255));
            cv::putText(display_,
                        std::to_string(bbox.azimuth()),
                        cv::Point(bbox.top_left_x(), bbox.top_left_y() + 30),
                        CV_FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 255));
        }
    }

    if (config_["visualization"]["show_projections"].getBool()) {
        // visualization of particle projections
        std::vector<feh::Vec2f> projections;
        GetProjection(projections, level);
        for (const auto &pt : projections) {
            cv::circle(display_, cv::Point(pt(0), pt(1)), 1, cv::Scalar(0, 0, 255), 2);
        }
    }

    if (config_["visualization"]["show_mean_projection"].getBool()) {
        auto mean_proj = ProjectMean(level);
        cv::circle(display_, cv::Point(mean_proj(0), mean_proj(1)), 1, cv::Scalar(0, 255, 0), 2);
    }

    if (config_["visualization"]["show_mean_boundary"].getBool()) {
        auto prediction = Render(level);
        OverlayMaskOnImage(prediction, display_, false, kColorMap.at(class_name_));
    } else if (config_["visualization"]["show_wireframe"].getBool()) {
        auto wireframe = RenderWireframe(level);
        OverlayMaskOnImage(wireframe, display_, true, kColorMap.at(class_name_));
    }

//    if (config_["visualization"]["show_correspondences"].getBool()) {
//        std::vector<EdgePixel> edgelist;
//        renderer->ComputeEdgePixels(MatForRender(mean_),
//                                    edgelist);
//        std::vector<OneDimSearchMatch> matches;
//        if (level > 0) {
//            oned_search_(edgelist, evidence_.back(), matches);
//        } else {
//            oned_search_(edgelist, evidence_.front(), matches);
//        }
//        for (const auto &match : matches) {
//            cv::line(display_,
//                     cv::Point(match.pt1_(0), match.pt1_(1)),
//                     cv::Point(match.pt2_(0), match.pt2_(1)),
//                     cv::Scalar(255, 0, 0),
//                     1);
//        }
//    }

    if (config_["visualization"]["show_evidence_as_thumbnail"].getBool()) {
        cv::Mat thumbnail;
        cv::resize(evidence_[0], thumbnail, cv::Size(cols_[0] / 4,
                                                     rows_[0] / 4));
        cv::cvtColor(thumbnail, thumbnail, CV_GRAY2RGB);
        cv::Rect rect(0, 0, thumbnail.cols, thumbnail.rows);
        thumbnail.copyTo(display_(rect));
    }
}

void Tracker::WriteOutParticles(const std::string &filename) const {
    particles_.WriteToFile(filename);
}

void Tracker::LogDebugInfo() {
    dbg_file_ << mean_.transpose() << "\n"; // << particles_.Covariance()
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// CONVENIENT RENDERING AND PROJECTION ROUTINES
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

Mat4f Tracker::MatForRender(const Vec4f &v) const{
    Vec4f s = StateFromLocalParam(v);
    // rotation around y axis in gravity aligned camera frame
    // where y axis of the camera frame is along the direction of gravity
    Eigen::AngleAxisf Ro(v(3), Vec3f::UnitY());
    Mat3f R = gwr_.rotationMatrix().transpose() * gwc0_.rotationMatrix() * Ro.toRotationMatrix();
//    Mat3f R = gwr_.rotationMatrix().transpose() * Ro.toRotationMatrix();

    Mat4f out;
    out(3, 3) = 1;
//    out.block<3, 3>(0, 0) =  R * Rg_.matrix().transpose();
    out.block<3, 3>(0, 0) = R * Rg_.matrix().transpose();
    out.block<3, 1>(0, 3) = s.head<3>();
    return out;
}

Mat4f Tracker::MatForRender() const{
    return MatForRender(mean_);
}

cv::Mat Tracker::Render(int level) {
    Sophus::SE3f pose(MatForRender(mean_));
    return RenderAt(pose, level);
}

cv::Mat Tracker::RenderAt(const Sophus::SE3f &object_pose,
                          int level) {
    auto renderer = renderers_[level];
    CHECK(renderer != nullptr);

    cv::Mat out(rows_[level], cols_[level], CV_8UC1);
    out.setTo(0);
    renderer->RenderEdge(object_pose.matrix(), out.data);
    return out;
}

cv::Mat Tracker::RenderWireframe(int level) {
    Sophus::SE3f pose(MatForRender(mean_));
    return RenderWireframeAt(pose, level);
}

cv::Mat Tracker::RenderMask(int level) {
    Sophus::SE3f pose(MatForRender(mean_));
    return RenderMaskAt(pose, level);
}

cv::Mat Tracker::RenderMaskAt(const Sophus::SE3f &object_pose, int level) {
    auto renderer = renderers_[level];
    CHECK(renderer != nullptr);

    cv::Mat out(rows_[level], cols_[level], CV_8UC1);
    out.setTo(0);
    renderer->RenderMask(object_pose.matrix(), out.data);
    return out;
}

cv::Mat Tracker::RenderWireframeAt(const Sophus::SE3f &object_pose,
                                   int level) {
    auto renderer = renderers_[level];
    CHECK(renderer != nullptr);

    cv::Mat out(rows_[level], cols_[level], CV_8UC1);
    out.setTo(0);
    renderer->RenderWireframe(object_pose.matrix(), out.data);
    return out;
}

cv::Mat Tracker::RenderDepth(int level) {
    Sophus::SE3f pose(MatForRender(mean_));
    return RenderDepthAt(pose, level);
}

cv::Mat Tracker::RenderDepthAt(const Sophus::SE3f &object_pose,
                               int level) {
    auto renderer = renderers_[level];
    CHECK(renderer != nullptr);

    cv::Mat out(rows_[level], cols_[level], CV_32FC1);
    out.setTo(0);
    renderer->RenderDepth(object_pose.matrix(), (float*) out.data);
    PrettyDepth(out);
    return out;
}


Vec2f Tracker::Project(const Vec3f &vertex,
                       int level) const {
    Vec2f xc = vertex.head<2>() / vertex(2);
    Vec2f xp(fx_[level] * xc(0) + cx_[level],
             fy_[level] * xc(1) + cy_[level]);
    return xp;
}

void Tracker::GetProjection(std::vector<Vec2f> &projections,
                            int level) const {
    projections.clear();
    for (int i = 0; i < particles_.size(); ++i) {
        if (particles_[i].shape_id() == best_shape_match_) {
            Vec3f v(particles_[i].v().head<3>());
            v(2) = std::exp(v(2));
            v.head<2>() *= v(2);
            v = gwc_.inverse() * gwr_ * v;
            projections.push_back(Project(v, level));
        }
    }
}

Vec2f Tracker::ProjectMean(int level) const {
    Vec3f v(mean_.head<3>());
    v(2) = std::exp(v(2));
    v.head<2>() *= v(2);
    v = gwc_.inverse() * gwr_ * v;
    return Project(v, level);
}

void Tracker::SwitchMeshForInference() {
    if (use_partial_mesh_) {
        for (auto &kv : shapes_) {
            Shape &s = kv.second;
            for (int i = 0; i < s.render_engines_.size(); ++i)
                if (!s.part_vertices_.empty()) s.render_engines_[i]->SetMesh(s.part_vertices_, s.part_faces_);
        }
    }   // no need to switch otherwise
}

void Tracker::SwitchMeshForVisualization() {
    if (use_partial_mesh_) {
        for (auto &kv : shapes_) {
            Shape &s = kv.second;
            for (int i = 0; i < s.render_engines_.size(); ++i)
                if (!s.vertices_.empty()) s.render_engines_[i]->SetMesh(s.vertices_, s.faces_);
        }
    } // no need to switch otherwise
}

void Tracker::ComputeQualityMeasure() {
    std::vector<EdgePixel> edgelist;
    renderers_.front()->OneDimSearch(MatForRender(), edgelist);
    std::array<float, 2> info;
    LogLikelihoodFromEdgelist(edgelist, &info);
    quality_.matching_ratio_ = info[0];
    quality_.mean_matching_distance_ = info[1];

    int wait_time = 15;
    CHECK_EQ(history_.size(), label_history_.size());
    if (history_.size() < wait_time) {
        quality_.since_last_label_change_ = -1;
        quality_.uncertainty_.setOnes();
    } else {
        for ( quality_.since_last_label_change_ = 1;
              quality_.since_last_label_change_ <= history_.size();
              ++quality_.since_last_label_change_) {
            if (label_history_[label_history_.size()-quality_.since_last_label_change_] != best_shape_match_) break;
        }
        std::vector<Vec4f> recent_hist{history_.end()-wait_time, history_.end()};

        Vec4f temporal_mean;
        for (const auto &v : recent_hist) {
            temporal_mean += v;
        }
        temporal_mean /= recent_hist.size();

        Mat4f cov;
        for (const auto &v : recent_hist) {
            Vec4f dv = v - temporal_mean;
            cov += dv * dv.transpose();
        }
        cov /= recent_hist.size();
//        std::cout << "temporal cov=\n" << cov << std::endl;
//        quality_.uncertainty_ <<  cov(0, 0) < 0.02 && cov(1, 1) < 0.02 && cov(2, 2) < 0.05 && cov(3, 3) < 0.5;
        quality_.uncertainty_ = cov.diagonal();
    }
}

bool Tracker::StationaryEnough() {
//    return true;
    return quality_.since_last_label_change_ > 10
        && (quality_.uncertainty_(0) < 0.02 && quality_.uncertainty_(1) < 0.02
            && quality_.uncertainty_(2) < 0.05 && quality_.uncertainty_(3) < 0.5);
}

bool Tracker::CloseEnough(float ratio_thresh, float dist_thresh, float score_thresh) {
    return quality_.matching_ratio_ > ratio_thresh
        && quality_.mean_matching_distance_ < dist_thresh
        && quality_.CNN_score_ > score_thresh;
}



}
}




