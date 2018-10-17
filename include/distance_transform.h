//
// Created by feixh on 10/25/17.
//
// reference:
// https://cs.brown.edu/~pff/papers/dt-final.pdf
#pragma once
#include <iostream>
#include <vector>
#include <limits>
#include <functional>

// 3rd party
#include "glog/logging.h"
#include "opencv2/core.hpp"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

namespace feh {

class DistanceTransform {
public:
    template<typename T>
    void operator()(const std::vector<T> &in, std::vector<T> &out, std::vector<int> &index_out) const {
        out.resize(in.size());
        index_out.resize(in.size());
        one_dim_distance_transform_internal(&in[0], in.size(), &out[0], &index_out[0]);
    }

    template<typename T>
    void operator()(const std::vector<T> &in, std::vector<T> &out) const {
        out.resize(in.size());
        one_dim_distance_transform_internal(&in[0], in.size(), &out[0], nullptr);
    }

    void operator()(const cv::Mat &in, cv::Mat &out) const {
        CHECK(in.type() == CV_32FC1);
        int rows = in.rows;
        int cols = in.cols;
        if (out.empty()) {
            out = cv::Mat(in.size(), in.type(), CV_32FC1);
        } else if (out.rows != rows || out.cols != cols
                   || out.type() != in.type() || out.channels() != 1) {
            LOG(WARNING) << "incompatible output mat";
            out = cv::Mat(in.size(), in.type(), CV_32FC1);
        }

        auto row_kernel = [&in, &cols, &out, this](const tbb::blocked_range<int> &range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                    this->one_dim_distance_transform_internal((float *) in.ptr(i),
                                                              cols,
                                                              (float *) out.ptr(i));
            }
        };
        tbb::parallel_for(tbb::blocked_range<int>(0, rows),
                          row_kernel,
                          tbb::auto_partitioner());

//        for (int i = 0; i < rows; ++i) {
//            one_dim_distance_transform_internal(
//                (float *) in.ptr(i),
//                cols,
//                (float *) out.ptr(i));
//        }

        auto col_kernel = [&out, &rows, this](const tbb::blocked_range<int> &range) {
            std::vector<float> tmp, out_tmp;
            tmp.resize(rows);
            out_tmp.resize(rows);
            for (int i = range.begin(); i < range.end(); ++i) {
                for (int j = 0; j < rows; ++j) tmp[j] = out.at<float>(j, i);
                this->operator()(tmp, out_tmp);
                for (int j = 0; j < rows; ++j) out.at<float>(j, i) = out_tmp[j];
            }
        };

        tbb::parallel_for(tbb::blocked_range<int>(0, cols),
                          col_kernel,
                          tbb::auto_partitioner());
    }

    void operator()(const cv::Mat &in, cv::Mat &out, cv::Mat &index) const {
        CHECK(in.type() == CV_32FC1);
        int rows = in.rows;
        int cols = in.cols;
        if (out.empty()) {
            out = cv::Mat(in.size(), in.type(), CV_32FC1);
        } else if (out.rows != rows || out.cols != cols
                   || out.type() != in.type() || out.channels() != 1) {
            LOG(WARNING) << "incompatible output mat";
            out = cv::Mat(in.size(), in.type(), CV_32FC1);
        }

        cv::Mat index_confined_to_row(index.size(), CV_32SC1);
        auto row_kernel = [&in, &cols, &out, &index_confined_to_row, this](const tbb::blocked_range<int> &range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                this->one_dim_distance_transform_internal((float *) in.ptr(i),
                                                          cols,
                                                          (float *) out.ptr(i),
                                                          (int *) index_confined_to_row.ptr(i));
            }
        };
        tbb::parallel_for(tbb::blocked_range<int>(0, rows),
                          row_kernel,
                          tbb::auto_partitioner());

        auto col_kernel = [&out, &rows, &index_confined_to_row, &index, this]
            (const tbb::blocked_range<int> &range) {
            std::vector<float> tmp, out_tmp;
            std::vector<int> index_tmp;
            tmp.resize(rows);
            out_tmp.resize(rows);
            index_tmp.resize(rows);
            for (int i = range.begin(); i < range.end(); ++i) {
                for (int j = 0; j < rows; ++j) tmp[j] = out.at<float>(j, i);
                this->operator()(tmp, out_tmp, index_tmp);
                for (int j = 0; j < rows; ++j) {
                    out.at<float>(j, i) = out_tmp[j];
                    index.at<cv::Vec2i>(j, i)(0) = index_confined_to_row.at<int>(index_tmp[j], i);
                    index.at<cv::Vec2i>(j, i)(1) = index_tmp[j];
                }
            }
        };

        tbb::parallel_for(tbb::blocked_range<int>(0, cols),
                          col_kernel,
                          tbb::auto_partitioner());
    }


private:
    /// \brief:
    template<typename T>
    void one_dim_distance_transform_internal(const T *data_ptr,
                                             int len,
                                             T *const out_ptr,
                                             int *const index_out_ptr = nullptr) const {
        std::vector<int> v(len, 0);
        std::vector<T> z(len + 1, 0);

        v[0] = 0;
        z[0] = std::numeric_limits<T>::lowest();
        z[1] = std::numeric_limits<T>::max();

        float s;    // intersection of parabolas
        for (int q = 1, k = 0; q < len;) {
            s = 0.5 * ((data_ptr[q] + q * q) - (data_ptr[v[k]] + v[k] * v[k])) / (q - v[k]);

            if (s <= z[k]) {
                --k;
            } else {
                ++k;
                v[k] = q;
                z[k] = s;
                z[k + 1] = std::numeric_limits<T>::max();
                ++q;
            }
        }

        for (int q = 0, k = 0; q < len; ++q) {
            while (z[k + 1] < q) ++k;
            out_ptr[q] = (q - v[k]) * (q - v[k]) + data_ptr[v[k]];
            if (index_out_ptr != nullptr) index_out_ptr[q] = v[k];
        }
    }

public:
    /// \brief: Convert single channel 8-bit image to floating point type.
    static
    cv::Mat Preprocess(const cv::Mat &mat) {
        cv::Mat out;
        mat.convertTo(out, CV_32FC1);
        out = cv::Scalar::all(255) - out;
        return out;
    }

    /// \brief: Convert floating type image to 8-bit image for visualization.
    static
    cv::Mat BuildView(const cv::Mat &img) {
        double min_value, max_value;
        cv::Point min_location, max_location;
        cv::minMaxLoc(img, &min_value, &max_value, &min_location, &max_location);
//        std::cout << "min @ (" << min_location << ")=" << min_value << "\n";
//        std::cout << "max @ (" << max_location << ")=" << max_value << "\n";

        cv::Mat display;
        cv::convertScaleAbs(img, display, 255.0 / (max_value - min_value), min_value);
        return display;
    }
};

}   //feh
