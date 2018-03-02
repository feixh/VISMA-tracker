//
// Created by feixh on 10/25/17.
//
// stl
#include <vector>
#include <memory>
#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include "time.h"
#include "math.h"

// 3rd party
#include "glog/logging.h"
#include "mpreal.h"
#include <Eigen/Dense>


namespace feh {

static const mpfr::mpreal kParticleMaxWeight = 1e10;

template <typename T, int DIM=4>
class Particle {
public:

    Particle():
        v_{0, 0, 0, 0},
        nw_(std::numeric_limits<double>::quiet_NaN()),
        isvalid_(true)
    {}

    Particle(const Eigen::Matrix<T, DIM, 1> &v, int sid=0, double log_w=0):
        v_(v),
        log_w_(log_w),
        w_(mpfr::exp(log_w_)),
        nw_(std::numeric_limits<double>::quiet_NaN()),
        isvalid_(true),
        shape_id_(sid)
    {}

    void set_edge_log_likelihood(double ll) { edge_log_likelihood_ = ll;}
    double edge_log_likelihood() const {
        return edge_log_likelihood_;
    }

    void set_log_w(double log_w) {
        log_w_ = log_w;
        w_ = mpfr::exp(log_w_);
        // FIXME: need to use best rounding method
        if (mpfr::isinf(w_)) w_ = kParticleMaxWeight;
        nw_ = std::numeric_limits<double>::quiet_NaN();
    }
    double log_w() const { return log_w_; }
    void set_zero_w() {
        w_ = 0;
        nw_ = std::numeric_limits<double>::quiet_NaN();
    }

    const Eigen::Matrix<T, DIM, 1> &v() const { return v_; }
    Eigen::Matrix<T, DIM, 1> &v() { return v_; }
    T v(int i) const { return v_(i); }
    const mpfr::mpreal &w() const { return w_; }
    void Perturbate(Eigen::Matrix<T, DIM, 1> const &perturbation) { v_ = v_ + perturbation; }
    double nw() const {
        CHECK(!std::isnan(nw_)) << "\033[91mnormalized weight is nan!!!\033[0m";
        return nw_;
    }
    void set_nw(double nw) { nw_ = nw; }
    bool IsNormalized() const { return std::isnan(nw_); }
    uint64_t id() const { return id_; }

    bool IsValid() const { return isvalid_; }
    void MakeInvalid() { isvalid_ = false; }
    void MakeValid() { isvalid_ = true; }

    int shape_id() const { return shape_id_; }
    void set_shape_id(int sid) { shape_id_ = sid; }


private:
    uint64_t id_;
    Eigen::Matrix<T, DIM, 1> v_;           // state vector
    double log_w_;  // log weight
    double edge_log_likelihood_;
    mpfr::mpreal w_;    // (un-normalized) weight
    double nw_;     // normalized weight
    bool isvalid_;  // auxiliary status
    int shape_id_;  // discrete random variable
};

template <typename T, int DIM = 4>
class Particles: public std::vector<Particle<T, DIM>> {
public:
    typedef Particle<T, DIM> ParticleType;
    typedef Eigen::Matrix<T, DIM, 1> StateType;

public:
    void Initialize(std::shared_ptr<std::knuth_b> generator=nullptr) {
        if (generator) {
            generator_ = generator;
        } else {
            generator_ = std::make_shared<std::knuth_b>(time(NULL));
        }
    }

    int MostProbableIndex();
    Eigen::Matrix<T, DIM, 1> Mean(int index = 0);

    Eigen::Matrix<T, DIM, DIM> Covariance();
    Eigen::Matrix<T, DIM, 1> Mode();

    bool Normalize();
    bool SystematicResampling();
    bool ResidualResampling();
    bool MultinomialResampling();
    int Subsample(int required_num_particles);

    // io
    void WriteToFile(const std::string &filename) const;
    /// \brief: Print value of particles with normalized weights.
    void Print() const;
    /// \brief: Print summary information of the ensemble of particles.
    void PrintSummary() const;

private:
    std::shared_ptr<std::knuth_b> generator_;
    bool is_normalized_ = false;
};

template <typename T, int DIM>
int Particles<T, DIM>::MostProbableIndex() {
    // compute marginal distribution over indices
    Normalize();
    std::unordered_map<int, double> prob;
    for (auto it = this->begin(); it != this->end(); ++it) {
        double val = prob[it->shape_id()];
        prob[it->shape_id()] = val + it->nw();
    }
//    // debug
//    for (auto key_val : prob) {
//        std::cout << "(" << key_val.first << "," << key_val.second << ")--";
//    }
//    std::cout << "\n";

    int most_prob_idx(-1);
    double most_prob(0);
    for (auto key_val : prob) {
        if (key_val.second > most_prob) {
            most_prob_idx = key_val.first;
            most_prob = key_val.second;
        }
    }
    return most_prob_idx;
}


// mean conditioned on index
template <typename T, int DIM>
Eigen::Matrix<T, DIM, 1> Particles<T, DIM>::Mean(int index) {
    Normalize();
    Eigen::Matrix<T, DIM, 1> out;
    double total_w(0);
    for (auto it = this->begin(); it != this->end(); ++it) {
        if (it->shape_id() == index) {
            out += it->nw() * it->v();
            total_w += it->nw();
        }
    }
    CHECK(std::isnormal(total_w));
    return out / total_w;
}

template <typename T, int DIM>
Eigen::Matrix<T, DIM, DIM> Particles<T, DIM>::Covariance() {
    Eigen::Matrix<T, DIM, 1> mean = Mean();
    Eigen::Matrix<T, DIM, DIM> cov;
    cov.setZero();
    for (auto it = this->begin(); it != this->end(); ++it) {
        Eigen::Matrix<T, DIM, 1> tmp((it->v() - mean).template cast<T>());
        cov += it->nw() * tmp * tmp.transpose();
    }
    return cov;
}

template <typename T, int DIM>
Eigen::Matrix<T, DIM, 1> Particles<T, DIM>::Mode() {
    Eigen::Matrix<T, DIM, 1> out;
    mpfr::mpreal max_w(0);
    for (auto it = this->begin(); it != this->end(); ++it) {
        if (it->w() > max_w) {
            max_w = it->w();
            out = it->v();
        }
    }
    return out;
}

template <typename T, int DIM>
bool Particles<T, DIM>::Normalize() {
//    if (is_normalized_) return;
    mpfr::mpreal w(0);
    w = std::accumulate(this->begin(), this->end(), mpfr::mpreal(0),
                        [](mpfr::mpreal const &x, ParticleType const &p) {
                            return x + p.w();
                        });
//    for (auto it = this->begin(); it != this->end(); ++it) {
//        std::cout << it->log_w() << ";;;" << it->w() << "\n";
//        CHECK(!mpfr::isinf(it->w())) << "inf w";
//        CHECK(!mpfr::isnan(it->w())) << "nan w";
//    }
//    std::cout << "weight sum=" << w << "\n";
    CHECK(!mpfr::isinf(w));
    CHECK(!mpfr::isnan(w));
    CHECK_GT(w, 0);
    for (auto it = this->begin(); it != this->end(); ++it) {
        double nw((it->w() / w).toDouble());
        it->set_nw(nw);
    }
//    is_normalized_ = true;
    return true;
}

template <typename T, int DIM>
bool Particles<T, DIM>::SystematicResampling() {
    for (int i = 0; i < this->size(); ++i) {
        if (!this->at(i).IsValid()) this->at(i).set_zero_w();
    }
    Normalize();
    int n(this->size());
    std::uniform_real_distribution<double> dist(0, 1.0f/n);
    double u1 = dist(*generator_);
    double uj = u1, inc = 1.0/n;

    double sum1(0), sum2;
    std::vector<int> counter(n, 0);
    for (int i = 0, j = 0; i < n; ++i) {
        sum2 = sum1 + this->at(i).nw();
//        std::cout << "i=" << i << " ;;; (" << sum1 << "," << sum2 << ")\n";
        if (std::isnan(sum2)) LOG(FATAL) << "got nan value";
        // check the interval between sum1 and sum2
        // where sum1 = sum of w up to index (i-1)
        // and sum2 = sum of w up to index i
        while (j < n && uj >= sum1 && uj <= sum2) {
            // every time we increase a counter by one
            // we also increases j by one, thus sum of all counters
            // should always equal to j, i.e., when j == n, we have
            // sum of all counters equal to n too.
            ++counter[i];
            ++j;
            uj += inc;
        }
        if (j < n && uj < sum1) {
            LOG(FATAL) << "something went wrong";
        }

        if (j < n && uj > sum2) {
            sum1 = sum2;
        }
        if (j >= n) break;
    }
//    std::cout << "uj=" << uj << std::endl;

    // debug
    int total_count(0);
    std::vector<ParticleType> original(this->begin(), this->end());
    for (int i = 0, j = 0; i < n; ++i) {
        for (int k = 0; k < counter[i]; ++k, ++j) {
            this->at(j) = original[i];
        }
        total_count += counter[i];
    }
//    CHECK_EQ(total_count, n);
    if (total_count < n) {
        LOG(WARNING) << "inconsitent particle number after resampling:"
                     << "need " << n << " got " << total_count << "\n";
        Print();
    }

    for (auto it = this->begin(); it != this->end(); ++it) {
        it->set_nw(1.0f / n);
        it->set_log_w(0);
    }

    return true;
}

//template <typename T>
//bool Particle<T>::ResidualResampling(Particles &particles)
//{}
//
//template <typename T>
//bool Particle<T>::MultinomialResampling(Particles &particles)
//{}

template <typename T, int DIM>
void Particles<T, DIM>::WriteToFile(const std::string &filename) const {
    std::ofstream out(filename, std::ios::out);
    CHECK(out.is_open()) << "failed to open file " << filename << "\n";
    for (auto it = this->begin(); it != this->end(); ++it) {
        out << it->v().transpose() << " " << it->nw() << "\n";
    }
    out.close();
}

template <typename T, int DIM>
void Particles<T, DIM>::Print() const {
    std::cout << "====================\n====================\n====================\n";
    std::cout << "= Particles\n";
    std::cout << "====================\n====================\n====================\n";
    for (auto it = this->begin(); it != this->end(); ++it) {
        std::cout << it->v().transpose() << " " << it->nw() << "\n";
    }
    std::cout << "\n";
}

template <typename T, int DIM>
void Particles<T, DIM>::PrintSummary() const {
    // count particles of each label
    std::unordered_map<int, int> index_counter;
    std::unordered_map<int, double> index_marginal;
    std::unordered_map<int, double> prob;
    for (auto it = this->begin(); it != this->end(); ++it) {
        if (index_marginal.count(it->shape_id())) {
            index_marginal[it->shape_id()] += it->nw();
            index_counter[it->shape_id()] += 1;
        } else {
            index_marginal[it->shape_id()] = 0;
            index_counter[it->shape_id()] = 0;
        }
    }

    // debug
    std::cout << "===== Particles Summary =====\n";
    std::cout << "total particles=" << this->size() << "\n";
    std::cout << "(index, #particle)=";
    for (auto key_val : index_counter) {
        std::cout << "(" << key_val.first << "," << key_val.second << ")";
    }
    std::cout << "\n";

    std::cout << "(index, marginal)=";
    for (auto key_val : index_marginal) {
        std::cout << "(" << key_val.first << "," << key_val.second << ")";
    }
    std::cout << "\n";

};

template <typename T, int DIM>
int Particles<T, DIM>::Subsample(int required_num_particles) {
    if (this->size() <= required_num_particles) return this->size();
//    std::random_shuffle(this->begin(), this->end());
    std::sort(this->begin(), this->end(), [](const ParticleType &p1, const ParticleType &p2) {
        return p1.log_w() > p2.log_w();
    });
    this->resize(required_num_particles);
    return required_num_particles;
}


}
