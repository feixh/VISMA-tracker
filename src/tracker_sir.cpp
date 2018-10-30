//
// Created by feixh on 10/29/17.
//
// Sample-Importance-Resample Filter

#include "tracker.h"

// system
#include <sys/stat.h>
#include <tracker.h>

// 3rd party
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "mpreal.h"

// own
#include "tracker_utils.h"

namespace feh {
namespace tracker {

float Tracker::LogLikelihoodFromEdgelist(const std::vector<EdgePixel> &edgelist,
                                         std::array<float, 2> *info,
                                         float match_ratio_order) {
    float total_dist(0);
    int matches(0);
    for (auto edgepixel : edgelist) {
        if (edgepixel.depth >= 0) {
            total_dist += edgepixel.depth;
            matches += 1;
        }
    }
    float match_ratio = matches / (edgelist.size() + eps);
    float average_match_distance = total_dist / (matches + eps);
    if (info) {
        (*info)[0] = match_ratio;
        (*info)[1] = average_match_distance;
    }
    return powf(match_ratio, match_ratio_order) / (average_match_distance + eps);
//    return 1.0 / (average_match_distance + eps);
}

void Tracker::PFUpdate(int level) {
    CHECK(status_ != TrackerStatus::OUT_OF_VIEW);
    if (level < 0) level = scale_level_ - 1;

    timer_.Tick("update");
    Particles<float, 4> saved_particles(particles_);
    // reset log probability
    for (auto &particle : particles_) {
        particle.set_log_w(0);
        particle.MakeValid();
    }
    int counter_invalid = ComputeProposals(level);
    ComputeLikelihood(level);
    ComputePrior(level);
    if (use_MC_move_) {
        MakeMonteCarloMove(level);
    }
    timer_.Tock("update");

    if (counter_invalid > particles_.size() / 2.0) {
        LOG(INFO) << TermColor::red << "use saved particles" << TermColor::endl;
        particles_ = saved_particles;
        saved_status_ = status_;
        status_ = TrackerStatus::OUT_OF_VIEW;
    } else {
        timer_.Tick("resampling");
        bool resample_ok = particles_.SystematicResampling();
        if (!resample_ok) {
            // TODO: all samples have zero weights, need to re-initialize
            LOG(INFO) << TermColor::yellow << "need to re-initialize" << TermColor::endl;
            particles_ = saved_particles;
        }
        timer_.Tock("resampling");
    }

    if(scale_level_ == 1
        || level == 0) {
        best_shape_match_ = particles_.MostProbableIndex();
        renderers_ = shapes_.at(best_shape_match_).render_engines_;
        mean_ = particles_.Mean(best_shape_match_);
        history_.push_back(mean_);
        label_history_.push_back(best_shape_match_);
        std::cout << image_fullpath_ << "\n";
        particles_.PrintSummary();
        LogDebugInfo();

    }

    // UPDATE POSE
    gwm_ = gwr_ * SE3(MatForRender());
}

int Tracker::ComputeProposals(int level) {
    if (level < 0) level = scale_level_ - 1;
    int invalid_counter(0);
    std::uniform_real_distribution<float> uniform_dist(0, 1);
    for (auto &particle : particles_) {
        Vec4f perturbation = RandomVector<4>(0, 1, generator_);
        particle.Perturbate(proposal_std_.cwiseProduct(perturbation));
        // mixed kernel for azimuth estimation
        if (uniform_dist(*generator_) < azi_uniform_mix_) {
            particle.v()[3] = uniform_dist(*generator_) * 2 * M_PI;
        }
        particle.v()[3] = WarpAngle(particle.v(3));

//        if (particle.v(2) < log(0.1) || particle.v(2) > log(5.0)) {
//            particle.MakeInvalid();
//            particle.set_zero_w();
//            ++invalid_counter;
//            continue;
//        } else
        {
            particle.MakeValid();
            double log_proposal(0);
            for (int i = 0; i < 3; ++i) {
                double tmp = perturbation(i);
                log_proposal += -(tmp * tmp * 0.5);
            }
            CHECK(std::isnormal(log_proposal)) << "abnormal log proposal value: pert=" << perturbation.transpose();
            log_proposal *= log_proposal_weight_[level];
            particle.set_log_w(particle.log_w() - log_proposal);
        }
    }
    return invalid_counter;
}

void Tracker::ComputeLikelihood(int level) {
    level = (level < 0 ? scale_level_-1 : level);

    RendererPtr renderer(nullptr);
//    const auto &evidence = evidence_[level];
//    const auto &evidence_dir = evidence_dir_[level];

    std::vector<cv::Rect> hyp_bbox_list;    // hypothesized bounding boxes
    std::uniform_real_distribution<float> uniform_dist(0, 1);
    for (auto &particle : particles_) {
#ifndef FEH_USE_MCMC_SHAPE_IDENTIFICATION
        if (uniform_dist(*generator_) < keep_id_prob_
            || shape_ids_.size() == 1) {
            // keep the current shape id
        } else {
            // perturbate shape id
            std::uniform_int_distribution<int> label_jump_dist(1, shape_ids_.size()-1);
            int new_shape_id = shape_ids_.at((particle.shape_id() + label_jump_dist(*generator_)) % shape_ids_.size());
            particle.set_shape_id(new_shape_id);
        }
#endif
        // pick proper render engine
        renderer = shapes_.at(particle.shape_id()).render_engines_[level];
        double log_likelihood;
        timer_.Tick("rendering");
        std::array<float, 6> score_and_corner;
        std::vector<EdgePixel> edgelist;
//        renderer->ComputeEdgePixels(MatForRender(particle.v()),
//                                    edgelist);
        renderer->OneDimSearch(MatForRender(particle.v()),
                               edgelist);
        timer_.Tock("rendering");

        if (use_CNN_) {
            cv::Rect rect = RectEnclosedByContour(edgelist,
                                                  renderer->rows(),
                                                  renderer->cols());
            // scale to match the input image size
            float ratio = rows_[0] / (float) renderer->rows();
            rect.x *= ratio;
            rect.y *= ratio;
            rect.width *= ratio;
            rect.height *= ratio;
            hyp_bbox_list.push_back(rect);
        }

        log_likelihood = LogLikelihoodFromEdgelist(edgelist);

#ifdef FEH_USE_MCMC_SHAPE_IDENTIFICATION
        ////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////
        // MONTE CARLO SHAPE IDENTIFICATION
        ////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////
        if (shape_ids_.size() > 1 && keep_id_prob_ < 1.0f) {
            if (uniform_dist(*generator_) > keep_id_prob_) {
                // random jum with probability keep_id_prob_
                std::uniform_int_distribution<int> label_jump_dist(1, shape_ids_.size()-1);
                int new_shape_id = shape_ids_.at((particle.shape_id() + label_jump_dist(*generator_)) % shape_ids_.size());
                CHECK_NE(new_shape_id, particle.shape_id());
                particle.set_shape_id(new_shape_id);

                double new_log_likelihood;
                std::array<float, 6> score_and_corner;
                std::vector<EdgePixel> edgelist;
                shapes_.at(new_shape_id).render_engines_[level]->OneDimSearch(MatForRender(particle.v()),
                                                                              edgelist);
                new_log_likelihood = LogLikelihoodFromEdgelist(edgelist);
                mpfr::mpreal new_l = mpfr::exp(new_log_likelihood);
                mpfr::mpreal old_l = mpfr::exp(log_likelihood);
                double accept_ratio = mpfr::min(1.0, new_l / old_l).toDouble();
                if (accept_ratio >= 1.0 ||
                    uniform_dist(*generator_) < accept_ratio) {
                    // accept
                    log_likelihood = new_log_likelihood;
                    particle.set_shape_id(new_shape_id);
                } else {
                    // do nothing;
                }
            }
        }   // END-OF-MONTE-CARLO-SHAPE-IDENTIFICATION
#endif


        particle.set_edge_log_likelihood(log_likelihood);
        log_likelihood *= log_likelihood_weight_[level];
        particle.set_log_w(particle.log_w() + log_likelihood);
    }

    // use CNN as an extra likelihood term
    if (use_CNN_) {
        if (level == 0) quality_.CNN_score_ = 0;
        // publish hypothesized bboxes so that Fast R-CNN can evaluate likelihood
        timer_.Tick("sending hypotheses");
        PublishBBoxProposals(hyp_bbox_list);
        timer_.Tock("sending hypotheses");

        timer_.Tick("hypothesis evaluation by NN");
        // wait messages likelihood messages
        bool message_received(false);
        while (!message_received) {
            if (port_->handle() == 0) {
                message_received = (handler_->tracker_id_ == id());
                if (message_received) {
                    DLOG(INFO) << "likelihood message received\n";
                    const auto &scores(handler_->scores_);
                    // now let's update particles with the second likelihood term
                    CHECK_EQ(particles_.size(), scores.size());
                    for (int i = 0; i < particles_.size(); ++i) {
                        auto &particle(particles_[i]);
                        double score = scores[i];
                        quality_.CNN_score_ += score;
//                if (score < CNN_prob_thresh_) {
//                    particle.set_zero_w();
//                    particle.MakeInvalid();
//                } else
                        {
                            double CNN_logL = CNN_log_likelihood_weight_[level] * std::log(score);
                            particle.set_log_w(particle.log_w() + CNN_logL);
                        }
                    }
                    quality_.CNN_score_ /= (scores.size() + eps);
                }
            }
        }
        timer_.Tock("hypothesis evaluation by NN");
    }
}

void Tracker::ComputePrior(int level) {
    if (level < 0) level = scale_level_ - 1;
//    if (convergence_counter_ > 0)
    {
        for (auto &particle : particles_) {
            if (particle.IsValid()) {
//                Vec3f dv(particle.v().head<3>() - init_state_.head<3>());
//                double log_prior(0);
//                for (int i = 0; i < 4; ++i) {
//                    double tmp = dv(i) / initial_std_(i);
//                    log_prior += -(tmp * tmp * 0.5);
//                }
                double log_prior = (M_PI/2 - particle.v(3));
                log_prior *= log_prior;
                log_prior *= log_prior_weight_[level];
                particle.set_log_w(particle.log_w() + log_prior);
            }
        }
    }
}

void Tracker::PublishBBoxProposals(const std::vector<cv::Rect> &rect_list) {
    vlslam_pb::BoundingBoxList bboxlist;
    CHECK(!image_fullpath_.empty()) << "image path is empty";
    char ss[256];
    sprintf(ss, "%04d%s", id(), image_fullpath_.c_str());
    bboxlist.set_description(ss);
    DLOG(INFO) << "image full path=" << image_fullpath_ << "\n";
    for (const auto &rect : rect_list) {
        auto bbox = bboxlist.add_bounding_boxes();
//            auto rect = this_pair.second;
        bbox->set_top_left_x(rect.x);
        bbox->set_top_left_y(rect.y);
        bbox->set_bottom_right_x(rect.x + rect.width);
        bbox->set_bottom_right_y(rect.y + rect.height);
        bbox->set_class_name(class_name_);
    }
    uint8_t *send_data = new uint8_t[bboxlist.ByteSize()];
    bboxlist.SerializeToArray(send_data, bboxlist.ByteSize());
    port_->publish("bbox", send_data, bboxlist.ByteSize());
    delete [] send_data;
    DLOG(INFO) << "LCM message with " << bboxlist.bounding_boxes_size() << " boxes sent\n";
}

void Tracker::MakeMonteCarloMove(int level) {
    level = (level < 0 ? scale_level_-1 : level);

    std::uniform_real_distribution<float> uni_dist(0, 1);

    int valid_counter(0);
    int moved_counter(0);
    for (auto &p : particles_) {
        if (p.IsValid()) {
            ++valid_counter;
            if (uni_dist(*generator_) > azi_flip_rate_) continue;
            float ca = WarpAngle( - p.v(3));  // complementary angle
            auto r = shapes_.at(p.shape_id()).render_engines_[level];
            std::vector<EdgePixel> edgelist;
            r->OneDimSearch(MatForRender({p.v(0), p.v(1), p.v(2), ca}), edgelist);
            float total_dist(0);
            int matches(0);
            for (auto edgepixel : edgelist) {
                if (edgepixel.depth >= 0) {
                    total_dist += edgepixel.depth;
                    matches += 1;
                }
            }
            float match_ratio = matches / (edgelist.size() + eps);
            float average_match_distance = total_dist / (matches + eps);
            double new_log_likelihood = match_ratio / (average_match_distance + eps);
            mpfr::mpreal new_l = mpfr::exp(new_log_likelihood);
            mpfr::mpreal old_l = mpfr::exp(p.edge_log_likelihood());
            double accept_rate = mpfr::min(1.0, new_l / old_l).toDouble();
            if (accept_rate >= 1.0
                || uni_dist(*generator_) < accept_rate) {
                // accept with probability of accept_rate

                // modify log likelihood
                p.set_log_w(p.log_w()
                                - log_likelihood_weight_[level] * p.edge_log_likelihood()
                                + log_likelihood_weight_[level] * new_log_likelihood);
                p.v()(3) = ca;  // change azimuth
                ++moved_counter;
//                std::cout << "new_l/old_l=" << new_l << "/" << old_l << "\n";
            }
        }
    }
    std::cout << TermColor::red << "MCMC move #" << moved_counter
              << "/" << valid_counter << TermColor::endl;
}


}   // namespace tracker
}   // namespace feh
