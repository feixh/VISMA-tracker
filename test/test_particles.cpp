//
// Created by visionlab on 10/26/17.
//
#include "particle.h"
#include "common/eigen_alias.h"

int main() {
    mpfr::mpreal::set_default_prec(mpfr::digits2bits(20));
    std::cout << "max=" << std::numeric_limits<mpfr::mpreal>::max() << "\n";
    std::cout << "exp(1000)=" << mpfr::exp(1000) << "\n";

    feh::Particles<float, 4> particles;
    particles.Initialize();
    particles.resize(10, {feh::Vec4f::Random(), 0, 0.0});

    int particle_counter(0);
    std::cout << "========== test constructor ==========\n";
    particle_counter = 0;
    for (auto const &p : particles) {
        std::cout << particle_counter++ << " ;;;" << p.v().transpose() << " ;;;" << p.w() << "\n";
    }

    std::cout << "========== test set ==========\n";
    particle_counter = 0;
    for (auto &p : particles) {
        p.set_log_w(0);
        std::cout << particle_counter++ << " ;;;" << p.v().transpose() << " ;;;" << p.w() << "\n";
    }

    std::cout << "========== test set big number ==========\n";
    particle_counter = 0;
    for (auto &p : particles) {
        p.set_log_w(1000);
        std::cout << particle_counter++ << " ;;;" << p.v().transpose() << " ;;;" << p.w() << "\n";
    }
    std::cout << "covariance=\n" << particles.Covariance();

    std::cout << "========== test mean with equal weights ==========\n";
    particle_counter = 0;
    for (auto &p : particles) {
        p.Perturbate(feh::Vec4f::Random());
        p.set_log_w(1000);
        std::cout << particle_counter++ << " ;;;" << p.v().transpose() << " ;;;" << p.w() << "\n";
    }
    std::cout << "mean=" << particles.Mean() << "\n";
    feh::Vec4f mean_p(0, 0, 0, 0);
    for (auto const &p : particles) {
        mean_p += p.v();
    }
    mean_p /= particles.size();
    std::cout << "true mean=" << mean_p << "\n";
    std::cout << "mode=" << particles.Mode() << "\n";
    std::cout << "true mode=" << "whichever is fine " << "\n";
    std::cout << "covariance=\n" << particles.Covariance();

    std::cout << "========== test mean with 1 dominant particle ==========\n";
    particle_counter = 0;
    for (auto &p : particles) {
        p.Perturbate(feh::Vec4f::Random());
        if (particle_counter == 0) {
            p.set_log_w(100);
        } else {
            p.set_log_w(0);
        }
        std::cout << particle_counter++ << " ;;;" << p.v().transpose() << " ;;;" << p.w() << "\n";
    }
    std::cout << "mean=" << particles.Mean() << "\n";
    std::cout << "true mean=" << particles[0].v() << "\n";
    std::cout << "mode=" << particles.Mode() << "\n";
    std::cout << "true mode=" << particles[0].v() << "\n";
    std::cout << "covariance=\n" << particles.Covariance();

    std::cout << "========== test resampling with 1 dominant particle ==========\n";
    particles.SystematicResampling();
    particle_counter = 0;
    for (auto &p : particles) {
        std::cout << particle_counter++ << " ;;;" << p.v().transpose() << " ;;;" << p.w() << "\n";
    }
    std::cout << "covariance=\n" << particles.Covariance();

//    feh::Particles<feh::Vec4f> particles;
//    particles.resize(100, {feh::Vec4f::Random(), 1.0f});
//    particles.Normalize();
//    for (const auto &p : particles) {
//        std::cout << p.w_ << " ";
//    }
//    std::cout << "\n";
//
//    std::cout << "==========\n==========\n==========\n";
//    std::cout << "exp(100)=" << exp(10.0);
//    particles.resize(100, {feh::Vec4f::Random(), exp(10.0)});
//    CHECK(particles.Normalize());
//    particles.PrintWeights();
//
//    std::cout << "==========\n==========\n==========\n";
//    std::cout << "exp(1000)=" << exp(1000.0);
//    particles.resize(100, {feh::Vec4f::Random(), exp(0100.0)});
//    CHECK(particles.Normalize());
//    particles.PrintWeights();

}

