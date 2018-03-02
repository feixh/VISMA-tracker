//
// Created by visionlab on 1/17/18.
//
#include "ukf.h"

#include <memory>
#include <fstream>

// 3rd party
//#include "matplotlibcpp.h"

using FloatType = double;
static constexpr int M = 2; // state dimension
static constexpr int N = 2; // measurement dimension
static constexpr FloatType dt = 0.01;
static constexpr FloatType freq = 2*M_PI;

//namespace plt = matplotlibcpp;

class MackeyGlassGenerator {
public:
    MackeyGlassGenerator(int n, FloatType cov_u, FloatType cov_v):
        n_(n),
        cov_u_(cov_u),
        cov_v_(cov_v){

        generator_ = std::make_shared<std::knuth_b>(time(NULL));
        std::normal_distribution<FloatType> dist;

        FloatType t(0);

        Eigen::Matrix<FloatType, M, 1> x0;
        x0.setRandom();
        x0[1] = cos(freq * t);
        x0[3] = sin(freq * t);
        x0_ = x0;
        for (int i = 0; i < n_; ++i, t += dt) {
//            auto last_x =  (i == 0? x0 : x_[i-1]);

            x_.push_back(Eigen::Matrix<FloatType, M, 1>::Zero());

            x_[i] << cos(freq * t) + sqrt(cov_u_) * dist(*generator_),
                sin(freq * t) + sqrt(cov_u_) * dist(*generator_);


            // noisy measurement
            Eigen::Matrix<FloatType, 2, 1> y;
            y << sqrt(x_[i][0]*x_[i][0] + x_[i][1]*x_[i][1]) + sqrt(cov_v_) * dist(*generator_),
                atan2(x_[i][1], x_[i][0]) + sqrt(cov_v_) * dist(*generator_);
            y_.push_back(y);
        }
    };

    void Print() {
        std::cout << "x\n";
        for (int i = 0; i < x_.size(); ++i) {
            std::cout << i << ":" << x_[i].transpose() << "\n";
        }
        std::cout << "y\n";
        for (int i = 0; i < y_.size(); ++i) {
            std::cout << i << ":" << y_[i] << "\n";
        }
    }


public:
    int n_;     // steps
    FloatType cov_u_, cov_v_; // covariance for process observation and noise
    std::shared_ptr< std::knuth_b > generator_;
    std::vector<Eigen::Matrix<FloatType, M, 1>> x_, xt_;
    Eigen::Matrix<FloatType, M, 1> x0_;
    std::vector<Eigen::Matrix<FloatType, N, 1>> y_;
};

using UKFType = feh::UKF<FloatType, 4, 2, 4, 2>;

UKFType::TypeX prop_func(const UKFType::TypeX &x,
                         const UKFType::TypeU &u) {
    UKFType::TypeX out;
    out[0] = x[0] + x[2] * dt;
    out[1] = x[1] + x[3] * dt;
    out[2] = x[2];
    out[3] = x[3];
    return out + u;
};

UKFType::TypeY pred_func(const UKFType::TypeX &x,
                         const UKFType::TypeV &v) {
    UKFType::TypeY out;
    out << sqrt(x[0]*x[0] + x[1]*x[1]), atan2(x[1], x[0]);
    out += v;
    return out;
};

int main(int argc, char **argv) {
    float cov_x(0.01), cov_u(0.01), cov_v(0.01);
    if (argc != 1) {
        if (argc == 4) {
            cov_x = std::atof(argv[1]);
            cov_x *= cov_x;

            cov_u = std::atof(argv[2]);
            cov_u *= cov_u;

            cov_v = std::atof(argv[3]);
            cov_v *= cov_v;
        } else {
            LOG(FATAL) << "could accept 1 or 4 parameters";
            exit(-1);
        }
    }
    MackeyGlassGenerator mg(1000, 0.01, 0.01);
//    mg.Print();
    UKFType ukf(1e-2, 2);    // with default parameters

    UKFType::TypeX x0;
    x0.setRandom();

    Eigen::Matrix<FloatType, UKFType::dim_x, UKFType::dim_x> P0;
    P0.setIdentity();
    P0 *= cov_x;

    Eigen::Matrix<FloatType, UKFType::dim_u, UKFType::dim_u> Pu;
    Pu.setIdentity();
    Pu *= cov_u;

    Eigen::Matrix<FloatType, UKFType::dim_v, UKFType::dim_v> Pv;
    Pv.setIdentity();
    Pv *= cov_v;

//    ukf.Initialize(mg.x0_, P0, Pu, Pv);
//    ukf.Initialize(mg.x0_ + 0.01 * Eigen::Matrix<FloatType, M, 1>::Random(), P0, Pu, Pv);
    ukf.Initialize(x0, P0, Pu, Pv);

    std::fstream out_x("x.out", std::ios::out);
    std::fstream out_xe("xe.out", std::ios::out);

    std::vector<float> x, xe, y, ye;

    if (out_x.is_open() && out_xe.is_open()) {
        for (int i = 0; i < mg.n_; ++i) {
            ukf.Update(prop_func, pred_func, mg.y_[i]);

            x.push_back(mg.x_[i](0));
            y.push_back(mg.y_[i](1));

            xe.push_back(ukf.Mean()(0));
            ye.push_back(ukf.Mean()(1));

            std::cout << "x=" << mg.x_[i].transpose() << "\n";
            std::cout << "xe=" << ukf.Mean().transpose() << "\n";

            out_x << mg.x_[i].transpose() << "\n";
            out_xe << ukf.Mean().transpose() << "\n";
        }
        out_x.close();
        out_xe.close();
    } else {
        LOG(FATAL) << "failed to open output files";
    }

//    {
//        plt::subplot(1, 2, 1);
//        plt::plot(x, "b");
//        plt::plot(xe, "r");
//        plt::subplot(1, 2, 2);
//        plt::plot(y, "b");
//        plt::plot(ye, "r");
//        plt::show();
//
//    }
}

