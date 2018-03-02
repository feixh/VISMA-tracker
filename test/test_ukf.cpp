//
// Created by visionlab on 1/16/18.
//
#include "gtest/gtest.h"

#define private public
#include "ukf.h"

constexpr int DIMX = 4;
constexpr int DIMY = 3;
constexpr int DIMU = 4;
constexpr int DIMV = 3;
constexpr int L = DIMX;
constexpr int N = (L << 1) + 1;


Eigen::Matrix<float, DIMX, 1> prop_func(const Eigen::Matrix<float, DIMX, 1> &X) {
    Eigen::Matrix<float, DIMX, 1> out;
    for (int i = 0; i < DIMX; ++i) {
        out[i] = X[i] + 1.0f;
    }
    return out;
};

Eigen::Matrix<float, DIMX, 1> noisy_prop_func(const Eigen::Matrix<float, DIMX, 1> &X,
                                              const Eigen::Matrix<float, DIMU, 1> &U) {
    Eigen::Matrix<float, DIMX, 1> out;
    for (int i = 0; i < DIMX; ++i) {
        out[i] = X[i] + U[i] + 1.0f;
    }
    return out;
};

Eigen::Matrix<float, DIMY, 1> pred_func(const Eigen::Matrix<float, DIMX, 1> &X) {
    Eigen::Matrix<float, DIMY, 1> out(0, 0, 0);
    for (int i = 0; i < DIMY; ++i) {
        out[i] = X[i] * 0.5f;
    }
    return out;
};

Eigen::Matrix<float, DIMY, 1> noisy_pred_func(const Eigen::Matrix<float, DIMX, 1> &X,
                                              const Eigen::Matrix<float, DIMV, 1> &V) {
    Eigen::Matrix<float, DIMY, 1> out(0, 0, 0);
    for (int i = 0; i < DIMY; ++i) {
        out[i] = X[i] * 0.5f + V[i];
    }
    return out;
};

namespace feh {

class UKFTest: public ::testing::Test {
public:
    UKFTest():
        ukf(1e-3, 2, 0.0)
    {
        Eigen::Matrix<float, DIMX, 1> X0;
        X0.setRandom();
        Eigen::Matrix<float, DIMX, DIMX> P0;
        P0.setRandom();
        P0 = P0 * P0;   // make it PSD
        Eigen::Matrix<float, DIMU, DIMU> Pu;
        Pu.setIdentity();
        Pu *= 0.01;

        Eigen::Matrix<float, DIMV, DIMV> Pv;
        Pv.setIdentity();
        Pv *= 0.01;
        ukf.Initialize(X0, P0, Pu, Pv);
    }

    UKF<float, DIMX, DIMY, DIMU, DIMV> ukf;
};

TEST_F(UKFTest, Initialize) {
//    ukf.Print();
    ASSERT_EQ(ukf.X_.size(), N);
    ASSERT_EQ(ukf.Y_.size(), N);
    for (int i = 1; i <= L; ++i) {
        ASSERT_TRUE(ukf.X_[0].isApprox(0.5*(ukf.X_[i] + ukf.X_[L+i])));
    }
}

TEST_F(UKFTest, Propagation) {
    // backup old values
    std::vector<decltype(ukf)::TypeX> Xold;
    for (const auto &X : ukf.X_) {
        Xold.push_back(X);
    }

    ukf.Propagate(prop_func);

    for (int i = 0; i < ukf.X_.size(); ++i) {
        ASSERT_EQ(ukf.X_[i].size(), DIMX);
        ASSERT_TRUE(ukf.X_[i].isApprox(Xold[i]+decltype(ukf)::TypeX::Constant(1)));
    }
}

TEST_F(UKFTest, NoisyPropagation) {
    // backup old values
    std::vector<decltype(ukf)::TypeX> Xold;
    for (const auto &X : ukf.X_) {
        Xold.push_back(X);
    }

    ukf.Propagate(noisy_prop_func);

    for (int i = 0; i < ukf.X_.size(); ++i) {
        ASSERT_EQ(ukf.X_[i].size(), DIMX);
        ASSERT_TRUE(ukf.X_[i].isApprox(Xold[i]+ ukf.U_[i] + decltype(ukf)::TypeX::Constant(1)));
    }
}

TEST_F(UKFTest, Prediction) {
    ukf.Predict(pred_func);
    for (int i = 0; i < ukf.Y_.size(); ++i) {
        ASSERT_EQ(ukf.Y_[i].size(), DIMY);
        ASSERT_TRUE(ukf.Y_[i].isApprox(ukf.X_[i].head<3>() * 0.5));
    }
}

TEST_F(UKFTest, NoisyPrediction) {
    ukf.Predict(noisy_pred_func);
    for (int i = 0; i < ukf.Y_.size(); ++i) {
        ASSERT_EQ(ukf.Y_[i].size(), DIMY);
        ASSERT_TRUE(ukf.Y_[i].isApprox(ukf.X_[i].head<3>() * 0.5 + ukf.V_[i]));
    }
}

}
