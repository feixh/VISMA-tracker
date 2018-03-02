//
// Created by feixh on 10/18/17.
//
#include <opencv2/imgproc.hpp>
#include "tracker_utils.h"

// 3rd party
#include "opencv2/highgui.hpp"
#include "tbb/parallel_for.h"
#include "glog/logging.h"

// own
#include "parallel_kernels.h"
#include "distance_transform.h"
#include "renderer.h"


namespace feh {

namespace tracker {
// utils
Mat4f Mat4FromState(const Vec4f &state) {
    Mat4f mat;
    mat(3, 3) = 1;
    mat.block<3, 3>(0, 0) = Eigen::AngleAxisf(state(3), Vec3f::UnitY()).toRotationMatrix();
    mat.block<3, 1>(0, 3) = state.head<3>();

    return mat;
}

Vec4f StateFromLocalParam(const Vec4f &local_param, Mat4f *jac) {
    Vec4f out;
    // inverse depth
//    out(2) = 1.0f / local_param(2);
    out(2) = std::exp(local_param(2));
    out(0) = local_param(0) * out(2);
    out(1) = local_param(1) * out(2);
    out(3) = local_param(3);

    if (jac) {
        throw;
        float x = local_param(0);
        float y = local_param(1);
        float z = local_param(2);
        float a = local_param(3);
        float z_inv = 1.0f / z;
        float z_inv2 = z_inv * z_inv;
        *jac << z_inv, 0, - x * z_inv2, 0,
            0, z_inv, -y * z_inv2, 0,
            0, 0, -z_inv2, 0,
            0, 0, 0, 1;
    }
    return out;
}

void NormalizeVertices(std::vector<float> &vertices) {
    int n(vertices.size() / 3);
    float sum[3] = {0, 0, 0};
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 3; ++j) {
            sum[j] += vertices[i * 3 + j];
        }
    }
    for (int i = 0; i < 3; ++i) sum[i] /= n;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 3; ++j) {
            vertices[i * 3 + j] -= sum[j];
        }
    }
}

void ScaleVertices(std::vector<float> &vertices, float scale_factor) {
    for (int i = 0; i < vertices.size(); ++i) {
        vertices[i] *= scale_factor;
    }
}

void RotateVertices(std::vector<float> &vertices, float angle) {
    Eigen::AngleAxisf aa(angle, Eigen::Vector3f::UnitY());
    int n(vertices.size() / 3);
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3f out = aa * Eigen::Map<Eigen::Vector3f>(&vertices[i * 3]);
        for (int j = 0; j < 3; ++j) {
            vertices[i * 3 + j] = out(j);
        }
    }
}

void FlipVertices(std::vector<float> &vertices) {
    int n(vertices.size() / 3);
    for (int i = 0; i < n; ++i) {
        // in CG coordinate system, y is pointing upwards and z is pointing toward us
        // while in CV coordinate system, y is pointing downwards and z is pointing forwards
        // Thus flip y and z axis.
        vertices[i * 3 + 1] = -vertices[i * 3 + 1];
        vertices[i * 3 + 2] = -vertices[i * 3 + 2];
    }
}

void OverlayMaskOnImage(const cv::Mat &mask,
                        cv::Mat &image,
                        bool invert_mask,
                        const uint8_t *color) {
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            if (mask.type() == CV_8U) {
                uint8_t value(mask.at<uchar>(i, j));
                if ((!invert_mask && value > 0)
                    || (invert_mask && value == 0)) {
                    if (color == nullptr) {
                        image.at<cv::Vec3b>(i, j) = cv::Vec3b(value, value, value);
                    } else {
                        image.at<cv::Vec3b>(i, j) = cv::Vec3b(color[0], color[1], color[2]);
                    }
                }
            } else if (mask.type() == CV_32F) {
                float value(mask.at<float>(i, j));
                if (value >= 0) {
                    if (color == nullptr) {
                        image.at<cv::Vec3b>(i, j)
                            = cv::Vec3b(value * 255,
                                        value * 255,
                                        value * 255);
                    } else {
                        image.at<cv::Vec3b>(i, j)
                            = cv::Vec3b(value * color[0],
                                        value * color[1],
                                        value * color[2]);
                    }
                }
            }
        }
    }
}

void PrettyDepth(cv::Mat &out, float z_near, float z_far) {
    // linearize depth
    tbb::parallel_for(tbb::blocked_range<int>(0, out.rows),
            [&out, &z_near, &z_far](const tbb::blocked_range<int> &range) {
                for (int i = range.begin(); i < range.end(); ++i) {
                    for (int j = 0; j < out.cols; ++j) {
                        float zb = out.at<float>(i, j);
                        if (zb > 0 && zb < 1) {
                            out.at<float>(i, j) = LinearizeDepth(zb, z_near, z_far) / z_far;
                        } else {
                            out.at<float>(i, j) = -1;
                        }
                    }
                }
            },
            tbb::auto_partitioner());
}

#define FEH_FLIP_AZIMUTH

int AzimuthIndexFromRadian(float rad) {
#ifdef FEH_FLIP_AZIMUTH
    int index = (int)floor(- rad / M_PI * 180);
#else
    int index = (int)floor(rad / M_PI * 180);
#endif
    if (index >= 360) {
        while (index >= 360) index -= 360;
    } else if (index < 0) {
        while (index < 0) index += 360;
    }
    CHECK_GE(index, 0);
    CHECK_LT(index, 360);
    return index;
}

float WarpAngle(float angle) {
    while (angle >= 2*M_PI) {
        angle -= 2*M_PI;
    }
    while (angle < 0) {
        angle += 2*M_PI;
    }
    return angle;
}
float RadianFromAzimuthIndex(int index) {
#ifdef FEH_FLIP_AZIMUTH
    float rad = 2 * M_PI - index / 180.0f * M_PI;
#else
    float rad = index / 180.0f * M_PI;
#endif
    return rad;
}

float BBoxArea(const vlslam_pb::BoundingBox &bbox) {
    return fabs((bbox.top_left_x() - bbox.bottom_right_x())
                * (bbox.top_left_y() - bbox.bottom_right_y()));
}

cv::Rect RectEnclosedByContour(const std::vector<EdgePixel> &edgelist, int rows, int cols) {
    if (!edgelist.empty()) {
        int min_x = 10000;
        int min_y = 10000;
        int max_x = 0;
        int max_y = 0;
        for (const auto &e : edgelist) {
            min_x = std::min<int>(min_x, e.x);
            min_y = std::min<int>(min_y, e.y);
            max_x = std::max<int>(max_x, e.x);
            max_y = std::max<int>(max_y, e.y);
        }
        min_x = std::max(0, min_x);
        min_y = std::max(0, min_y);
        max_x = std::min(cols-1, max_x);
        max_y = std::min(rows-1, max_y);
        return cv::Rect(cv::Point(min_x, min_y), cv::Point(max_x, max_y));
    } else {
        return cv::Rect(cv::Point(0, 0), cv::Point(0, 0));
    }
}

float ComputeIoU(const cv::Rect &r1, const cv::Rect &r2) {
    cv::Rect intersection = r1 & r2;
    cv::Rect enclosing_rect = r1 | r2;
    cv::Point offset = enclosing_rect.tl();
    cv::Mat tmp(enclosing_rect.height, enclosing_rect.width, CV_8UC1);
    tmp.setTo(0);
    cv::rectangle(tmp, r1 - offset, 1, -1);
    cv::rectangle(tmp, r2 - offset, 1, -1);
    return intersection.area() / (cv::sum(tmp)[0] + eps);
}

cv::Rect InflateRect(const cv::Rect &rect, int rows, int cols, int pad) {
    if (pad == 0 ) return rect;
    cv::Rect out(cv::Point(std::max(rect.x - pad, 0),
                           std::max(rect.y - pad, 0)),
                 cv::Point(std::min(rect.x + rect.width + pad, cols),
                           std::min(rect.y + rect.height + pad, rows)));
    return out;
}

void ComputeColorHistograms(const cv::Mat &image,
                            const cv::Rect &bbox,
                            cv::Rect &inflated_bbox,
                            std::vector<VecXf> &histf,
                            std::vector<VecXf> &histb,
                            int histogram_size,
                            int inflate_size) {

    // clean up histograms
    histf.resize(3);
    histb.resize(3);
    for (int i = 0; i < 3; ++i) {
        histf[i].setZero(histogram_size);
        histb[i].setZero(histogram_size);
    }

    cv::Mat patch;
    image(bbox).copyTo(patch);
    for (int i = 0; i < patch.rows; ++i) {
        for (int j = 0; j < patch.cols; ++j) {
            const cv::Vec3b &c = patch.at<cv::Vec3b>(i, j);
            for (int k = 0; k < 3; ++k) {
                histf[k](floor(c(k) / (256/histogram_size))) += 1;
            }
        }
    }
    inflated_bbox = cv::Rect(
        cv::Point(std::max(0, bbox.x - inflate_size),
                  std::max(0, bbox.y - inflate_size)),
        cv::Point(std::min(bbox.x + bbox.width + inflate_size, image.cols),
                  std::min(bbox.y + bbox.height + inflate_size, image.rows)));
    image(inflated_bbox).copyTo(patch);
    for (int i = 0; i < patch.rows; ++i) {
        for (int j = 0; j < patch.cols; ++j) {
            const cv::Vec3b &c = patch.at<cv::Vec3b>(i, j);
            for (int k = 0; k < 3; ++k) {
                histb[k](floor(c(k) / (256/histogram_size))) += 1;
            }
        }
    }
    for (int k = 0; k < 3; ++k) {
        histb[k] -= histf[k];
        histb[k] /= (histb[k].sum() + 1e-4);
        histf[k] /= (histf[k].sum() + 1e-4);
    }


//#ifdef FEH_PWP_DEBUG
//    std::cout << TermColor::bold << TermColor::white << "foreground color histogram" << TermColor::endl;
//    std::cout << TermColor::blue << histf[0].transpose() << TermColor::endl;
//    std::cout << TermColor::green << histf[1].transpose() << TermColor::endl;
//    std::cout << TermColor::red << histf[2].transpose() << TermColor::endl;
//
//    std::cout << TermColor::bold << TermColor::white << "background color histogram" << TermColor::endl;
//    std::cout << TermColor::blue << histb[0].transpose() << TermColor::endl;
//    std::cout << TermColor::green << histb[1].transpose() << TermColor::endl;
//    std::cout << TermColor::red << histb[2].transpose() << TermColor::endl;
//
//    for (int i = 0; i <3; ++i) {
//        plt::plot(std::vector<float>{histf[i].data(), histf[i].data()+histf[i].size()}, hist_code_[i] + "o");
//        plt::plot(std::vector<float>{histb[i].data(), histb[i].data()+histb[i].size()}, hist_code_[i] + "+");
//        plt::show();
//    }
//#endif

}


void ComputeColorHistograms(const cv::Mat &image,
                            const cv::Mat &mask,
                            cv::Rect &bbox,
                            std::vector<VecXf> &histf,
                            std::vector<VecXf> &histb,
                            int histogram_size,
                            int inflate_size) {
    // reserve space and clean up
    histf.resize(3);
    histb.resize(3);
    for (int i = 0; i < 3; ++i) {
        histf[i].setZero(histogram_size);
        histb[i].setZero(histogram_size);
    }
    // compute the color histograms of the foreground (inside projection mask)
    int top_left_x(image.cols), top_left_y(image.rows), bottom_right_x(0), bottom_right_y(0);
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            if (mask.at<uint8_t>(i, j) == 0) {
                // projection mask
                const cv::Vec3b &c = image.at<cv::Vec3b>(i, j);
                for (int k = 0; k < 3; ++k) {
                    int index = floor(c(k) / (256 / histogram_size));
                    histf[k](index) += 1;
                }
                if (i < top_left_y) top_left_y = i;
                if (i > bottom_right_y) bottom_right_y = i;
                if (j < top_left_x) top_left_x = j;
                if (j > bottom_right_x) bottom_right_x = j;
            }
        }
    }

    // now compute histograms of the inflated bounding box
    bbox = cv::Rect(cv::Point(std::max(0, top_left_x - inflate_size),
                              std::max(0, top_left_y - inflate_size)),
                    cv::Point(std::min(image.cols, bottom_right_x + inflate_size),
                              std::min(image.rows, bottom_right_y + inflate_size)));
    for (int i = bbox.y; i < bbox.y + bbox.height; ++i) {
        for (int j = bbox.x; j < bbox.x + bbox.width; ++j) {
            if(mask.at<uint8_t>(i, j) > 0) {
                const cv::Vec3b &c = image.at<cv::Vec3b>(i, j);
                for (int k = 0; k < 3; ++k) {
                    int index = floor(c(k) / (256 / histogram_size));
                    histb[k](index) += 1;
                }
            }
        }
    }

    for (int k = 0; k < 3; ++k) {
        histb[k] /= (histb[k].sum() + 1e-4);
        histf[k] /= (histf[k].sum() + 1e-4);
    }
}


void ComputePixelwisePosterior(const cv::Mat &image,
                               std::vector<cv::Mat> &P,
                               std::vector<VecXf> &hist_f,
                               std::vector<VecXf> &hist_b) {
    ComputePixelwisePosterior(image, P, hist_f, hist_b,
                              cv::Rect(0, 0, image.cols, image.rows));
}

void ComputePixelwisePosterior(const cv::Mat &image,
                               std::vector<cv::Mat> &P,
                               std::vector<VecXf> &hist_f,
                               std::vector<VecXf> &hist_b,
                               const cv::Rect &bbox) {
//    timer_.Tick("pixelwise posterior");
    float area_f = 500, area_b = 100;
    P.resize(2);
    P[0] = cv::Mat(image.rows, image.cols, CV_32FC1);   // slice 0 for foreground
    P[1] = cv::Mat(image.rows, image.cols, CV_32FC1);   // slice 1 for background
    P[0].setTo(0);
    P[1].setTo(1.0 / area_b);
    tbb::parallel_for(tbb::blocked_range<int>(bbox.y, bbox.y + bbox.height),
                      PixelwisePosteriorKernel(image,
                                               hist_f,
                                               hist_b,
                                               area_f,
                                               area_b,
                                               P,
                                               bbox),
                      tbb::auto_partitioner());
//    timer_.Tock("pixelwise posterior");
//    std::vector<cv::Mat> pfpb(2);
//    cv::split(P, pfpb);
    const auto &pfpb = P;
    cv::Mat pf_display = DistanceTransform::BuildView(pfpb[0]);
    cv::Mat pb_display = DistanceTransform::BuildView(pfpb[1]);
    cv::imshow("Pf", pf_display);
    cv::imshow("Pb", pb_display);
}

cv::Mat LinearizeDepthMap(const cv::Mat &zbuffer, float z_near, float z_far) {
    cv::Mat out(zbuffer.size(), CV_32FC1);
    tbb::parallel_for(tbb::blocked_range<int>(0, zbuffer.rows),
    [&out, &zbuffer, z_near, z_far](const tbb::blocked_range<int> &range) {
        for (int i = range.begin(); i < range.end(); ++i) {
            for (int j = 0; j < zbuffer.cols; ++j) {
                out.at<float>(i, j) = LinearizeDepth(zbuffer.at<float>(i, j), z_near, z_far);
            }
        }
    });
    return out;
}

cv::Mat PrettyZBuffer(const cv::Mat &zbuffer, float z_near, float z_far) {
    double min_val, max_val;
    cv::minMaxIdx(zbuffer, &min_val, &max_val);
    cv::Mat out;
    zbuffer.convertTo(out, CV_8UC1, 255 / (max_val-min_val), -min_val);
    return out;
}

cv::Mat PrettyLabelMap(const cv::Mat &label_map, const std::vector<std::array<uint8_t, 3>> &color_map) {
    cv::Mat out(label_map.size(), CV_8UC3);
    tbb::parallel_for(tbb::blocked_range<int>(0, label_map.rows),
    [&out, &label_map, &color_map](const tbb::blocked_range<int> &range) {
        for (int i = range.begin(); i < range.end(); ++i) {
            for (int j = 0; j < label_map.cols; ++j) {
                int idx = label_map.at<int32_t>(i, j) + 1;  // -1: background
                out.at<cv::Vec3b>(i, j) = {color_map[idx][0], color_map[idx][1], color_map[idx][2]};
            }
        }
    });
    return out;
}


}   // namespace tracker

}   // namespace feh
