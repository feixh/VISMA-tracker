//
// Created by visionlab on 1/30/18.
//
#include "tracker.h"
#include "tracker_utils.h"

namespace feh {
namespace tracker {

void Tracker::InitializeFromBoundingBox(const vlslam_pb::BoundingBox &bbox,
                                        const Sophus::SE3f &cam_pose,
                                        const Sophus::SO3f &Rg,
                                        std::string imagepath) {

    image_fullpath_ = imagepath;
    class_name_ = bbox.class_name();
    LOG(INFO) << "object pose initialized";

    gwr_ = cam_pose;
    Rg_ = Rg;

    for (auto sid : shape_ids_) {
        for (auto r: shapes_.at(sid).render_engines_) {
            r->SetCamera(grc_.inverse().matrix());
        }
    }
    status_ = TrackerStatus::INITIALIZING;

    float center_x = 0.5f * (bbox.top_left_x() + bbox.bottom_right_x());
    float center_y = 0.5f * (bbox.top_left_y() + bbox.bottom_right_y());
    init_state_(0) = (center_x - cx_[0]) / fx_[0];
    init_state_(1) = (center_y - cy_[0]) / fy_[0];
    init_state_(2) = log(config_["filter"]["initial_depth"].asDouble());
    init_state_(3) = 0;
    mean_ = init_state_;

    // initialize particles
    particles_.resize(max_num_particles_, {mean_, 0, 0.0f});
    // diffuse pose
    std::uniform_real_distribution<float> uniform_azimuth_dist(0, M_PI*2);
    for (auto &particle : particles_) {
        Vec4f perturbation = RandomVector<4>(0, 1.0, generator_);
        particle.Perturbate(initial_std_.cwiseProduct(perturbation));
        particle.v()(3) = WarpAngle(uniform_azimuth_dist(*generator_));    // initialize from uniform distribution

//        if (particle.v(2) < log(0.1) || particle.v(2) > log(5.0)) {
//            particle.set_zero_w();
//        } else
        {
            double log_w(0);
//            for (int i = 0; i < 3; ++i) {
//                float tmp = perturbation(i);
//                log_w += -(tmp * tmp * 0.5);
//            }
            particle.set_log_w(log_w);
        }
    }
    // assign same initial pose to each possible shape
    for (int i = 1; i < shape_ids_.size(); ++i) {
        for (int j = 0; j < max_num_particles_; ++j) {
            particles_.push_back(particles_[j]);
        }
    }
    // assign equally likely shape labels
    std::vector<int> tmp_ids;
    for (int sid : shape_ids_) {
        tmp_ids.insert(tmp_ids.begin(), max_num_particles_, sid);
    }
//    std::shuffle(tmp_ids.begin(), tmp_ids.end(), *generator_);
    for (int i = 0; i < particles_.size(); ++i) {
        particles_[i].set_shape_id(tmp_ids[i]);
        CHECK(shapes_.count(particles_[i].shape_id()));
    }


    particles_.SystematicResampling();
    mean_ = particles_.Mean();
    std::cout << "mean(0)=" << mean_.transpose() << "\n";
    particles_.PrintSummary();
}

}
}
