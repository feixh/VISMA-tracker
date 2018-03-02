//
// Created by feixh on 11/13/17.
//
// kernels for CPU parallelization
#pragma once

namespace feh {

namespace tracker {

/// \brief: Given an image of floating type, binarize it.
/// In the case of a depth map from OpenGL depth buffer, mark z value > 0.99 (background)
/// with 255, mark the rest with 0.
template<typename T>
class BinarizeKernel {
public:
    BinarizeKernel(const cv::Mat &image, cv::Mat &mask, T threshold) :
        image_(image),
        mask_(mask),
        threshold_(threshold),
        rect_(0, 0, image.cols, image.rows) {}

    BinarizeKernel(const cv::Mat &image, cv::Mat &mask, T threshold, const cv::Rect &rect) :
        image_(image),
        mask_(mask),
        threshold_(threshold),
        rect_(rect) {}

    void operator()(const tbb::blocked_range<int> &range) const {
        for (int i = range.begin(); i < range.end(); ++i) {
            for (int j = rect_.x; j < rect_.x + rect_.width; ++j) {
                mask_.at<uint8_t>(i, j) = image_.at<T>(i, j) > threshold_ ? 255 : 0;
            }
        }
    }

private:
    const cv::Mat &image_;
    cv::Mat &mask_;
    int cols_;
    T threshold_;
    cv::Rect rect_;
};

/// \brief: Detect edges. Edge pixels are marked with 0, background or non-edge foreground
/// pixels are marked with parameter kBigNumber
const static float kBigNumber = 1e9;

template<typename T>
class EdgeDetectionKernel {
public:
    EdgeDetectionKernel(const cv::Mat &image, cv::Mat &edge, float threshold) :
        image_(image),
        edge_(edge),
        threshold_(threshold),
        rect_(0, 0, image.cols, image.rows) {}

    EdgeDetectionKernel(const cv::Mat &image, cv::Mat &edge, float threshold,
                        const cv::Rect &rect) :
        image_(image),
        edge_(edge),
        threshold_(threshold),
        rect_(rect) {}

    void operator()(const tbb::blocked_range<int> &range) const {
        float value[9];
        for (int i = range.begin(); i < range.end(); ++i) {
            for (int j = rect_.x; j < rect_.x + rect_.width; ++j) {
                if (image_.at<T>(i, j) > 0.99) {
                    // ensure only pixels within the mask are classified as edge pixels
                    edge_.at<float>(i, j) = kBigNumber;
                    continue;
                }
                int counter(0);
                for (int ii = std::max(0, i - 1); ii <= std::min(image_.rows - 1, i + 1); ++ii) {
                    for (int jj = std::max(0, j - 1); jj <= std::min(image_.cols - 1, j + 1); ++jj) {
//                        value[counter] = image_.at<T>(ii, jj);
                        value[counter] = LinearizeDepth(image_.at<T>(ii, jj),
                                                        0.05f, 5.0f);
                        ++counter;
                    }
                }
                if (counter < 9) {
                    edge_.at<float>(i, j) = kBigNumber;
                } else {
                    float delta = 0.25 * (fabs(value[1] - value[7])
                                          + fabs(value[5] - value[3])
                                          + fabs(value[0] - value[8])
                                          + fabs(value[2] - value[6]));
                    edge_.at<float>(i, j) = delta > threshold_ ? 0 : kBigNumber;
                }
            }
        }
    }

private:
    const cv::Mat &image_;
    cv::Mat &edge_;
    float threshold_;
    cv::Rect rect_;
};

/// \brief: Given a segmentation mask and a distance field
class SignedDistanceKernel {
public:
    SignedDistanceKernel(const cv::Mat &mask,
                         const cv::Mat &distance_field,
                         cv::Mat &signed_distance_field) :
        mask_(mask),
        distance_(distance_field),
        signed_distance_(signed_distance_field),
        rect_(0, 0, mask.cols, mask.rows) {}

    SignedDistanceKernel(const cv::Mat &mask,
                         const cv::Mat &distance_field,
                         cv::Mat &signed_distance_field,
                         const cv::Rect &rect) :
        mask_(mask),
        distance_(distance_field),
        signed_distance_(signed_distance_field),
        rect_(rect) {}

    void operator()(const tbb::blocked_range<int> &range) const {
        for (int i = range.begin(); i < range.end(); ++i) {
            for (int j = 0; j < mask_.cols; ++j) {
                // foreground pixels are marked with 0
                // and foreground have minus sign
                float sign = mask_.at<uint8_t>(i, j) > 0 ? 1 : -1;
                signed_distance_.at<float>(i, j) = sign * distance_.at<float>(i, j);
            }
        }
    }

private:
    const cv::Mat &mask_;
    const cv::Mat &distance_;
    cv::Mat &signed_distance_;
    cv::Rect rect_;
};

/// brief: Compute spatial derivative of the image using central difference.
class CentralDifferenceKernel {
public:
    CentralDifferenceKernel(const cv::Mat &image, cv::Mat &dxdy) :
        image_(image),
        dxdy_(dxdy) {}

    void operator()(const tbb::blocked_range<int> &range) const {
        for (int i = range.begin(); i < range.end(); ++i) {
            for (int j = 0; j < image_.cols; ++j) {
                if (i > 0 && i < image_.rows - 1
                    && j > 0 && j < image_.cols - 1) {
                    cv::Vec2f &value = dxdy_.at<cv::Vec2f>(i, j);
                    value(0) = 0.5 * (image_.at<float>(i, j + 1)
                                      - image_.at<float>(i, j - 1));
                    value(1) = 0.5 * (image_.at<float>(i + 1, j)
                                      - image_.at<float>(i - 1, j));
                } else {
                    dxdy_.at<cv::Vec2f>(i, j) = {0, 0};
                }
            }
        }
    }

private:
    const cv::Mat &image_;
    cv::Mat &dxdy_;
};

static const float pi_inv = 1.0 / M_PI;

/// brief: Evaluate heaviside field given a signed distance field.
class HeavisideKernel {
public:
    HeavisideKernel(const cv::Mat &signed_distance,
                    cv::Mat &heaviside,
                    float reach=1.0f) :
        signed_distance_(signed_distance),
        heaviside_(heaviside),
        rect_(0, 0, signed_distance.cols, signed_distance.rows),
        reach_of_smooth_heaviside_(reach){}

    HeavisideKernel(const cv::Mat &signed_distance,
                    cv::Mat &heaviside,
                    const cv::Rect &rect,
                    float reach=1.0f) :
        signed_distance_(signed_distance),
        heaviside_(heaviside),
        rect_(rect),
        reach_of_smooth_heaviside_(reach){}

    void operator()(const tbb::blocked_range<int> &range) const {
        for (int i = range.begin(); i < range.end(); ++i) {
            for (int j = rect_.x; j < rect_.x + rect_.width; ++j) {
                float dh_dsdf;
                cv::Vec2f &value = heaviside_.at<cv::Vec2f>(i, j);
                float dy_dx(0);
                value(0) = smooth_heaviside_func(
                    signed_distance_.at<float>(i, j),
                    reach_of_smooth_heaviside_,
                    &dy_dx);
                value(1) = dy_dx;
            }
        }
    }

    template<typename T>
    static
    T smooth_heaviside_func(T x, T b = 1.0, T *dy_dx = nullptr) {
        T y = -pi_inv * atan(b * x) + 0.5;
        if (dy_dx) {
            T bx2(b * x);
            bx2 *= bx2;
            *dy_dx = -pi_inv * b / (1 + bx2);
        }
        return y;
    }



private:
    const cv::Mat &signed_distance_;
    cv::Mat &heaviside_;
    cv::Rect rect_;
    float reach_of_smooth_heaviside_;
};

/// brief: Evaluate pixel-wise posterior given the image,
/// model (histograms) and inflated bounding box.
class PixelwisePosteriorKernel {
public:
    PixelwisePosteriorKernel(const cv::Mat &image,
                             const std::vector<VecXf> &histf,
                             const std::vector<VecXf> &histb,
                             float area_f,
                             float area_b,
                             cv::InputArray &P,
                             const cv::Rect &rect) :
        image_(image),
        hist_f_(histf),
        hist_b_(histb),
        P_(P),
        is_single_mat_(P.kind() == cv::_InputArray::MAT),
        rect_(rect),
        area_f_(area_f),
        area_b_(area_b),
        histogram_size_(histf[0].size()) {}

    PixelwisePosteriorKernel(const cv::Mat &image,
                             const std::vector<VecXf> &histf,
                             const std::vector<VecXf> &histb,
                             float area_f,
                             float area_b,
                             cv::InputArray &P) :
        image_(image),
        hist_f_(histf),
        hist_b_(histb),
        P_(P),
        is_single_mat_(P.kind() == cv::_InputArray::MAT),
        rect_(0, 0, image.cols, image.rows),
        area_f_(area_f),
        area_b_(area_b),
        histogram_size_(histf[0].size()) {}

    void operator()(const tbb::blocked_range<int> &range) const {
        for (int i = range.begin(); i < range.end(); ++i) {
            for (int j = rect_.x; j < rect_.x + rect_.width; ++j) {
                const cv::Vec3b &c = image_.at<cv::Vec3b>(i, j);
                float pf(1), pb(1);
                for (int k = 0; k < 3; ++k) {
                    int index_l = floor(c(k) / (256 / histogram_size_));
                    int index_r = round(c(k) / (256 / histogram_size_));
                    index_r = std::min(index_r, histogram_size_-1);
                    pf *= (hist_f_[k](index_l) + hist_f_[k](index_r)) * 0.5;
                    pb *= (hist_b_[k](index_l) + hist_b_[k](index_r)) * 0.5;
                }

//                pf = area_f_ * pf / (area_f_ * pf + area_b_ * pb);
//                pb = area_b_ * pb / (area_f_ * pf + area_b_ * pb);
                pf = pf / (area_f_ * pf + area_b_ * pb);
                pb = pb / (area_f_ * pf + area_b_ * pb);
                if (is_single_mat_) {
                    cv::Vec2f &value = P_.getMat().at<cv::Vec2f>(i, j);
                    value(0) = pf;
                    value(1) = pb;
                } else {
                    P_.getMat(0).at<float>(i, j) = pf;
                    P_.getMat(1).at<float>(i, j) = pb;
                }
            }
        }
    }

private:
    const cv::Mat &image_;
    const std::vector<VecXf> &hist_f_;
    const std::vector<VecXf> &hist_b_;
    cv::InputArray &P_;
    bool is_single_mat_;
    cv::Rect rect_;
    float area_f_, area_b_;
    int histogram_size_;

};

}   // namespace tracker

}   // namespace feh

