//
// Created by visionlab on 1/18/18.
//
#include "eigen_alias.h"
#include "renderer.h"
#include "particle.h"

namespace feh {

/// \brief: Pose Initializer
/// \param bbox: [x1, y1, x2, y2] where (x1, y1) is top left corner; (x2, y2) is bottom right corner
void Initialize(const Eigen::Vec4f &bbox,
                const Eigen::Vec4f &init_std,
                RendererPtr renderer,
                int max_num_particles) {
    float center_x = 0.5f * (bbox[0] + bbox[2]);
    float center_y = 0.5f * (bbox[1] + bbox[3]);
    Eigen::Vec4f mean;
    mean << (center_x - renderer->cx()) / renderer->fx(),
        (center_y - renderer->cy()) / renderer->fy(),
        1.f / 2.0,
        0;

    // initialize depth
    std::vector<EdgePixel> edgelist;
//    renderer_->ComputeEdgePixels(MatForRender(mean_), edgelist);
//    cv::Rect rect = RectEnclosedByContour(edgelist, renderer_->rows(), renderer_->cols());
//    float ratio = sqrt(rect.area() / (BBoxArea(bbox) * scale_factor_ * scale_factor_));
//    if (std::isnormal(ratio)) mean_(2) /= ratio;


    Particles<float, 4> particles;
    // initialize particles
    particles.resize(1000, {mean_, 0.0f});
    int counter(0);
    std::uniform_real_distribution<float> uni_dist(0, 2*M_PI);
    decltype(particles)::ParticleType best_particle;
    for (auto &particle : particles) {
        Vec4f perturbation = RandomVector<4>(0, 1.0, generator);
        particle.Perturbate(init_std.cwiseProduct(perturbation));
        particle.v()(3) = uni_dist(generator);

        // compute IoU
        renderer->ComputeEdgePixels(MatForRender(particle.v()), edgelist);
        cv::Rect rect = RectEnclosedByContour(edgelist, renderer->rows(), renderer->cols());
        if (rect.area()) {
        }

        if () {
            best_particle = particle;
        }
    }
}

}   // namespace feh
