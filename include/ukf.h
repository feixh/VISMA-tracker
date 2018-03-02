//
// Created by visionlab on 1/16/18.
//
#pragma once

// stl
#include <vector>
#include <iostream>
#include <array>
#include <functional>

// 3rd party
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include "glog/logging.h"

#define FEH_DEBUG_UKF

namespace feh {

// Unscented Kalman Filter
// reference:
//  https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter
// T: type of elements, float or double
// DIM_X: dimension of state
// DIM_Y: dimension of observation
template <typename T, int DIM_X, int DIM_Y, int DIM_U=DIM_X, int DIM_V=DIM_Y>
class UKF {
public:
    using TypeX = Eigen::Matrix<T, DIM_X, 1>;
    using TypeY = Eigen::Matrix<T, DIM_Y, 1>;
    using TypeU = Eigen::Matrix<T, DIM_U, 1>;
    using TypeV = Eigen::Matrix<T, DIM_V, 1>;

    using TypeXU = Eigen::Matrix<T, DIM_X+DIM_U, 1>;
    using TypeXV = Eigen::Matrix<T, DIM_X+DIM_V, 1>;

    static constexpr int DIM_XU = DIM_X + DIM_U;
    static constexpr int DIM_XV = DIM_X + DIM_V;

    static constexpr int dim_x = DIM_X;
    static constexpr int dim_y = DIM_Y;
    static constexpr int dim_u = DIM_U;
    static constexpr int dim_v = DIM_V;


    // Propagation and prediction with noises.
    using Propagation = std::function<TypeX(const TypeX&, const TypeU&)>;
    using Prediction = std::function<TypeY(const TypeX&, const TypeV&)>;

    /// \brief: Constructor.
    /// \param a: Primary scaling parameter, which determines the spread of the sigma points around
    /// the mean and usually is set to a small positive value (e.g. 1e-3).
    /// \param k: Secondary scaling parameter, usually set to 0.
    /// \param b: Used to incorporate prior knowledge of the distribution of X. For Gaussian distribution, b=2 is optimal.
    UKF(T a=1e-3, T b=2, T k=0):
        a_(a), b_(b), k_(k) {
    }

    /// \brief: Initialize filter state given initial mean and covariance.
    /// \param init_mean: Initial mean of state X.
    /// \param init_cov: Initial covaraince of state X.
    /// \param cov_u: Covaraince of process noise u.
    /// \param cov_v: Covariance of observation noise v.
    void Initialize(const TypeX &init_mean,
                    const Eigen::Matrix<T, DIM_X, DIM_X> &init_cov,
                    const Eigen::Matrix<T, DIM_U, DIM_U> &cov_u,
                    const Eigen::Matrix<T, DIM_V, DIM_V> &cov_v) {
        Xm_ = init_mean;
        Px_ = init_cov;
        Pu_ = cov_u;
        Pv_ = cov_v;
    }

    /// \brief: Generate Sigma Points given mean and covariance of the distribution.
    template <int DIM>
    void GenerateSigmaPoints(const Eigen::Matrix<T, DIM, 1> &mean,
                             const Eigen::Matrix<T, DIM, DIM> &cov,
                             std::array<Eigen::Matrix<T, DIM, 1>, 2*DIM+1> &sigma_points) const {
        sigma_points[0] = mean;
        auto scaled_P = cov * (DIM + lambda(DIM));
        auto sqrt_scaled_P = scaled_P.sqrt();
        for (int i = 1; i <= DIM; ++i) {
            sigma_points[i]   = sigma_points[0] + sqrt_scaled_P.col(i-1);
            sigma_points[DIM+i] = sigma_points[0] - sqrt_scaled_P.col(i-1);
        }
    }

    /// \brief Update function public interface
    /// \param prop_func: propagation function, takes state and control noise sigma points.
    /// \param pred_func: prediction function, takes state and measurement noise sigma points.
    void Update(Propagation prop_func, Prediction pred_func, const TypeY &Y);

    void Print() {
//        for (int i = 0; i < N; ++i) {
//            std::cout << "x[" << i << "]:" << X_[i].transpose() << "\n";
//        }
//        std::cout << "P=\n" << Px_ << "\n";
//        std::cout << "Py=\n" << Py_ << "\n";
//        std::cout << "Pxy=\n" << Pxy_ << "\n";
//        std::cout << "wm=";
//        for (int i = 0; i < Wm_.size(); ++i) std::cout << Wm_[i] << " ";
//        std::cout << "\n";
//
//        std::cout << "wc=";
//        for (int i = 0; i < Wc_.size(); ++i) std::cout << Wc_[i] << " ";
//        std::cout << "\n";
    }

    // accessors for debugging
    TypeX Mean() const {
        return Xm_;
    }

private:
    /// \brief Propagate distribution over state via sigma points
    /// \param prop_func: propagation function. usually f in filtering literature
    void Propagate(Propagation prop_func);

    /// \brief Predict observation after transformation of state sigma points.
    /// \param pred_func: prediction function. usually h in filtering literature
    void Predict(Prediction pred_func);

    /// \brief Update state given actual measurement.
    /// \param Y: measurement
    void Update(const TypeY &Y);

    T lambda(int L) const {
        return a_ * a_ * (L+k_) - L;
    }
    T Wm(int i, int L) const {
        T l(lambda(L));
        return (i == 0 ? l / (L+l) : 0.5 / (L+l));
    }
    T Wc(int i, int L) const {
        T l(lambda(L));
        return (i == 0 ? l / (L+l) + (1-a_*a_+b_) : 0.5 / (L+l));
    }


private:
    std::array<TypeXU, 2*DIM_XU+1> Xa_xu_;    // augmented state X+U
    std::array<TypeX, 2*DIM_XU+1> X_; // state sigma points

    std::array<TypeXV, 2*DIM_XV+1> Xa_xv_;    // augmented state X+V
    std::array<TypeY, 2*DIM_XV+1> Y_;   // prediction sigma points

    TypeX Xm_; // X mean
    TypeY Ym_; // Y mean
    Eigen::Matrix<T, DIM_X, DIM_X> Px_;  // Cov(x, x)
    Eigen::Matrix<T, DIM_Y, DIM_Y> Py_; // Cov(y, y)
    Eigen::Matrix<T, DIM_X, DIM_Y> Pxy_;    // Cov(x, y)
    Eigen::Matrix<T, DIM_X, DIM_Y> K_;  // kalman gain
    Eigen::Matrix<T, DIM_U, DIM_U> Pu_; // Cov(u, u)
    Eigen::Matrix<T, DIM_V, DIM_V> Pv_; // Cov(v, v)

    T a_, b_, k_;   // parameters alpha, beta, kappa, and lambda
};

template <typename T, int DIM_X, int DIM_Y, int DIM_U, int DIM_V>
void UKF<T, DIM_X, DIM_Y, DIM_U, DIM_V>::Propagate(Propagation prop_func) {
    // construct mean of augmented state
    static TypeXU Xa;
    Xa.setZero();
    Xa.template head<DIM_X>() = Xm_;

    // construct covariance of augmented state
    static Eigen::Matrix<T, DIM_XU, DIM_XU> Pa;
    Pa.setZero();
    Pa.template block<DIM_X, DIM_X>(0, 0) = Px_;
    Pa.template block<DIM_U, DIM_U>(DIM_X, DIM_X) = Pu_;

    GenerateSigmaPoints(Xa, Pa, Xa_xu_);    // sigma points of augmented state

    // update mean
    Xm_.setZero();
    for (int i = 0; i < Xa_xu_.size(); ++i) {
        const TypeXU &Xa(Xa_xu_[i]);
        X_[i] = prop_func(Xa.template head<DIM_X>(), Xa.template tail<DIM_U>());
        Xm_ += Wm(i, DIM_XU) * X_[i];
#ifdef FEH_DEBUG_UKF
        std::cout << "X[" << i << "]=" << X_[i].transpose() << "\n";
#endif
    }

    // update covariance
    Px_.setZero();
    for (int i = 0; i < X_.size(); ++i) {
        auto dX = X_[i] - Xm_;
        Px_ += Wc(i, DIM_XU) * dX * dX.transpose();
    }
};

template <typename T, int DIM_X, int DIM_Y, int DIM_U, int DIM_V>
void UKF<T, DIM_X, DIM_Y, DIM_U, DIM_V>::Predict(Prediction pred_func) {
    // construct mean of augmented state
    static TypeXV Xa;
    Xa.setZero();
    Xa.template head<DIM_X>() = Xm_;

    // construct covariance of augmented state
    static Eigen::Matrix<T, DIM_XV, DIM_XV> Pa;
    Pa.setZero();
    Pa.template block<DIM_X, DIM_X>(0, 0) = Px_;
    Pa.template block<DIM_V, DIM_V>(DIM_X, DIM_X) = Pv_;

    // generate sigma points
    GenerateSigmaPoints(Xa, Pa, Xa_xv_);

    Ym_.setZero();
    for (int i = 0; i < Xa_xv_.size(); ++i) {
        const TypeXV &Xa(Xa_xv_[i]);
        Y_[i] = pred_func(Xa.template head<DIM_X>(), Xa.template tail<DIM_V>());
        Ym_ += Wm(i, DIM_XV) * Y_[i];

#ifdef FEH_DEBUG_UKF
        std::cout << "Y[" << i << "]=" << Y_[i].transpose() << "\n";
#endif
    }

    Py_.setZero();
    for (int i = 0; i < Y_.size(); ++i) {
        auto dY = Y_[i] - Ym_;
        Py_ += Wc(i, DIM_XV) * dY * dY.transpose();
    }
};

template <typename T, int DIM_X, int DIM_Y, int DIM_U, int DIM_V>
void UKF<T, DIM_X, DIM_Y, DIM_U, DIM_V>::Update(const TypeY &Y) {
    Pxy_.setZero();
    for (int i = 0; i < Xa_xv_.size(); ++i) {
        Pxy_ += Wc(i, DIM_XV)
            * (Xa_xv_[i].template head<DIM_X>() - Xm_) * (Y_[i] - Ym_).transpose();
    }
    K_.transpose() = Py_.llt().solve(Pxy_.transpose());

#ifdef FEH_DEBUG_UKF
    {
        Eigen::EigenSolver<Eigen::Matrix<T, DIM_X, DIM_X>> esx(Px_);
        std::cout << "eigs(Px) BEFORE update=" << esx.eigenvalues().transpose() << "\n";

        Eigen::EigenSolver<Eigen::Matrix<T, DIM_Y, DIM_Y>> esy(Py_);
        std::cout << "eigs(Py)=" << esy.eigenvalues().transpose() << "\n";
    };
#endif

    Xm_ += K_ * (Y - Ym_);
    Px_ -= K_ * Py_ * K_.transpose();

#ifdef FEH_DEBUG_UKF
    {
        Eigen::EigenSolver<Eigen::Matrix<T, DIM_X, DIM_X>> esx(Px_);
        std::cout << "eigs(Px) AFTER update=" << esx.eigenvalues().transpose() << "\n";
    };
#endif

}

template <typename T, int DIM_X, int DIM_Y, int DIM_U, int DIM_V>
void UKF<T, DIM_X, DIM_Y, DIM_U, DIM_V>::Update(Propagation prop_func,
                                                Prediction pred_func,
                                                const TypeY &Y) {
#ifdef FEH_DEBUG_UKF
    std::cout << "y in=" << Y.transpose() << "\n";
#endif
    Propagate(prop_func);
    Predict(pred_func);
    Update(Y);
};

}
