//
// Created by feixh on 10/24/17.
//
#include "oned_search.h"
#include "bresenham.h"

// 3rd party
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "glog/logging.h"
#include "tbb/parallel_for.h"

namespace feh {

/// \brief: 1-dimensional search along normal of edge pixels.
void OneDimSearch::operator()(const std::vector<EdgePixel> &in_edgelist,
                              const cv::Mat &target,
                              std::vector<OneDimSearchMatch> &matches,
                              const cv::InputArray &direction_array) const
{

//    std::random_shuffle(edgelist.begin(), edgelist.end());
    std::vector<EdgePixel> edgelist(in_edgelist);
    std::sort(edgelist.begin(), edgelist.end(),
              [&](const EdgePixel &e1, const EdgePixel &e2) {
                  return (e1.x < e2.x) || (e1.x == e2.x && e1.y < e2.y);
              });

    if (parallel_) {
        Parallel(edgelist, target, matches, direction_array);
        return;
    }


    bool use_dir = (direction_array.kind() != cv::_InputArray::STD_VECTOR) && direction_consistency_thresh_ > 0;
    cv::Mat target_dir;
    if (use_dir) target_dir = direction_array.getMat(0);
    // parameters

    matches.clear();
    int search_counter(0);
    for (int i = 0; i < edgelist.size(); i += step_length_) {
        ++search_counter;
        cv::Point uv0(edgelist[i].x, edgelist[i].y);
        cv::Point best_match;
        bool found_match(false);

        float theta = edgelist[i].dir;
        float cos_th = cos(theta);
        float sin_th = sin(theta);
        if (cos_th < 0) {
            cos_th = -cos_th;
            sin_th = -sin_th;
        }
        cv::Point uv1(uv0.x + search_line_length_ * cos_th,
                      uv0.y + search_line_length_ * sin_th);
        cv::Point uv2(uv0.x - search_line_length_ * cos_th,
                      uv0.y - search_line_length_ * sin_th);

        cv::LineIterator it1(target, uv0, uv1);
        for (int j = 0; j < it1.count; ++j, ++it1) {
            if (*(*it1) > intensity_threshold_
                && (!use_dir
                    || fabs(cos(theta - target_dir.at<float>(it1.pos())))
                       > direction_consistency_thresh_) ) {
                    best_match = it1.pos();
                    found_match = true;
                    break;
                }
        }

        // FIXME: implement my own bresenham line iterator, should be correct and efficient
        // reference:
        // https://github.com/ssloy/tinyrenderer
        // and wiki:

//        {
//            uv1.x = std::min(std::max(0, uv1.x), target.cols-1);
//            uv1.y = std::min(std::max(0, uv1.y), target.rows-1);
//            float delta_x = uv1.x - uv0.x;
//            float delta_y = uv1.y - uv0.y;
//            float delta_err = std::min(fabs(delta_y / delta_x), 10.0);
//            float err = 0;
//            for (int x = uv0.x, y = uv0.y; x <= uv1.x; ++x) {
//                if (target.at<uint8_t>(y, x) > intensity_threshold_) {
//                    if (!use_dir ||
//                        fabs(cos(theta - target_dir.at<float>(y, x))) > direction_consistency_thresh_) {
//                        best_match.x = x;
//                        best_match.y = y;
//                        found_match = true;
//                        break;
//                    }
//                }
//                err += delta_err;
//                while (err >= 0.5) {
//                    y += (delta_y > 0 ? 1 : -1);
//                    err -= 1;
//                }
//            }
//        }
//        std::cout << "loop 1 done\n";

        if (found_match) {
            float len = cv::norm(best_match - uv0);
            uv2 = cv::Point(uv0.x - len * cos_th,
                            uv0.y - len * sin_th);
        }
        cv::LineIterator it2(target, uv0, uv2);
        for (int j = 0; j < it2.count; ++j, ++it2) {
            if (*(*it2) > intensity_threshold_
                && (!use_dir ||
                    fabs(cos(theta - target_dir.at<float>(it2.pos())))
                    > direction_consistency_thresh_) ) {
                best_match = it2.pos();
                found_match = true;
                break;
            }
        }
//        {
//            uv2.x = std::min(std::max(0, uv2.x), target.cols-1);
//            uv2.y = std::min(std::max(0, uv2.y), target.rows-1);
//            float delta_x = uv2.x - uv0.x;
//            float delta_y = uv2.y - uv0.y;
//            float delta_err = std::min(fabs(delta_y / delta_x), 10.0);
//            float err = 0;
//            for (int x = uv0.x, y = uv0.y; x >= uv2.x; --x) {
//                if (target.at<uint8_t>(y, x) > intensity_threshold_) {
//                    if (!use_dir ||
//                        fabs(cos(theta - target_dir.at<float>(y, x))) > direction_consistency_thresh_) {
//                        best_match.x = x;
//                        best_match.y = y;
//                        found_match = true;
//                        break;
//                    }
//                }
//                err += delta_err;
//                while (err >= 0.5) {
//                    y += (delta_y > 0 ? 1 : -1);
//                    err -= 1;
//                }
//            }
//        }
//        std::cout << "loop 2 done\n";

        if (found_match) {
            OneDimSearchMatch match(uv0.x, uv0.y, best_match.x, best_match.y);
            match.Set(edgelist[i]);
            matches.push_back(std::move(match));
        }

    }
//    LOG(INFO) << "found #" << matches.size() << " matches among"
//              << "#" << search_counter << " searched pixels";
}

void OneDimSearch::BuildMatchView(const cv::Mat &ref_img,
                    const cv::Mat &target_img,
                    const std::vector<OneDimSearchMatch> &matches,
                    cv::Mat &out) {
    int rows = ref_img.rows;
    int cols = ref_img.cols;
    int offset = 0; //rows;
    out = cv::Mat(rows + offset, cols, CV_8UC3);
    if (ref_img.channels() == 1) {
        cv::cvtColor(ref_img, out(cv::Rect(0, 0, cols, rows)), CV_GRAY2RGB);
    } else {
        if (ref_img.type() != CV_8UC3) {
            cv::convertScaleAbs(ref_img,
                                out(cv::Rect(0, 0, cols, rows)),
                                255.0f, 0.0f);
            std::cout << "convert to 8UC3\n";
        } else {
            ref_img.copyTo(out(cv::Rect(0, 0, cols, rows)));
        }
    }
    cv::Mat tmp;
    if (offset == 0) {
        cv::cvtColor(target_img, tmp, CV_GRAY2RGB);
        cv::addWeighted(out, 1.0, tmp, 1.0, 0.0, out);
    } else {
        cv::cvtColor(target_img, out(cv::Rect(0, offset, cols, rows)), CV_GRAY2RGB);
    }

    for (const auto &match: matches) {
        auto &pt1(match.pt1_);
        auto &pt2(match.pt2_);

        cv::Mat display = out.clone();
        cv::line(display, cv::Point(pt1(0), pt1(1)), cv::Point(pt2(0), pt2(1) + offset), cv::Scalar(0, 0, 255), 2.0);
        cv::imshow("match view", display);
        char c = cv::waitKey();
        if (c == 'q') break;
    }
    for (const auto &match: matches) {
        auto &pt1(match.pt1_);
        auto &pt2(match.pt2_);

        cv::line(out, cv::Point(pt1(0), pt1(1)), cv::Point(pt2(0), pt2(1) + offset), cv::Scalar(0, 0, 255), 2.0);
    }

}

/// \brief: Visualize normals with colors (like optical flow visualization).
void OneDimSearch::BuildNormalView(const std::vector<EdgePixel> &edgelist_,
                     cv::Mat &normal_view) {
    // reference color wheel:
    // https://www.pyngl.ucar.edu/Examples/Images/color3.2.png
    cv::Mat _hsv[3], hsv;
    _hsv[0] = cv::Mat::zeros(normal_view.size(), CV_32F);
    _hsv[1] = cv::Mat::ones(normal_view.size(), CV_32F);
    _hsv[2] = cv::Mat::zeros(normal_view.size(), CV_32F);

    for (const auto &edgepixel: edgelist_) {
        _hsv[0].at<float>(int(edgepixel.y), int(edgepixel.x)) = edgepixel.dir / M_PI * 180;
        _hsv[2].at<float>(int(edgepixel.y), int(edgepixel.x)) = 1.0;
    }
    std::cout << "#edge pixels = " << edgelist_.size() << "\n";
    cv::merge(_hsv, 3, hsv);

    //convert to BGR and show
    cv::Mat search_direction_rgb;
    cv::cvtColor(hsv, search_direction_rgb, cv::COLOR_HSV2RGB);
//    normal_view = cv::Mat(search_direction_rgb.size(), CV_8UC3);
    cv::convertScaleAbs(search_direction_rgb,
                        normal_view,
                        255.0f, 0.0f);
}

/// \brief: Visualize normals with short line segments.
void OneDimSearch::BuildNormalView2(const std::vector<EdgePixel> &in_edgelist_,
                      cv::Mat &normal_view) {

    std::vector<EdgePixel> edgelist_(in_edgelist_);

    std::sort(edgelist_.begin(), edgelist_.end(),
              [&](const EdgePixel &e1, const EdgePixel &e2) {
                  return (e1.x < e2.x) || (e1.x == e2.x && e1.y < e2.y);
              });

//    std::random_shuffle(edgelist_.begin(), edgelist_.end());

//    normal_view = cv::Mat(kRows, kCols, CV_8UC3);
    CHECK_EQ(normal_view.channels(), 3);

    normal_view.setTo(cv::Scalar(0, 0, 0));

    int step_length = 5;
    int search_line_length = 10;
    for (int i = 0; i < edgelist_.size(); i += step_length) {
        cv::Point uv0(edgelist_[i].x, edgelist_[i].y);

        float theta = edgelist_[i].dir;
        float cos_th = cos(theta);
        float sin_th = sin(theta);
        cv::Point uv1(uv0.x + search_line_length * cos_th,
                      uv0.y + search_line_length * sin_th);
        cv::Point uv2(uv0.x - search_line_length * cos_th,
                      uv0.y - search_line_length * sin_th);

        cv::circle(normal_view, uv0, 1, cv::Scalar(0, 0, 255), 1);
        cv::line(normal_view, uv0, uv1, cv::Scalar(0, 255, 0), 1);
        cv::line(normal_view, uv0, uv2, cv::Scalar(0, 255, 255), 1);
    }

}

void OneDimSearch::Parallel(const std::vector<EdgePixel> &edgelist,
                            const cv::Mat &target,
                            std::vector<OneDimSearchMatch> &matches,
                            const cv::InputArray &direction_array) const {
    // allocate enough space first
    std::vector<OneDimSearchMatch> match_buffer;
    match_buffer.resize(edgelist.size());

    bool use_dir = (direction_array.kind() != cv::_InputArray::STD_VECTOR) && direction_consistency_thresh_ > 0;
    cv::Mat target_dir;
    if (use_dir) target_dir = direction_array.getMat(0);

    auto kernel = [&edgelist, &target, &target_dir, &use_dir, &match_buffer, this](const tbb::blocked_range<size_t> &range) {
        int search_counter(0);
        for (size_t i = range.begin(); i < range.end(); i += step_length_) {
            // parameters
            ++search_counter;
            cv::Point uv0(edgelist[i].x, edgelist[i].y);
            cv::Point best_match;
            bool found_match(false);

            int cols = target.cols;
            int rows = target.rows;

            float theta = edgelist[i].dir;
            float cos_th = cos(theta);
            float sin_th = sin(theta);

#define FEH_USE_OPENCV_LINEITERATOR
#ifdef FEH_USE_OPENCV_LINEITERATOR
            cv::Point uv1{uv0.x+search_line_length_*cos_th, uv0.y+search_line_length_*sin_th};
            cv::LineIterator it1(target, uv0, uv1);
            for (int j = 0; j < it1.count; ++j, ++it1) {
                if (*(*it1) > intensity_threshold_
                    && (!use_dir ||
                        fabs(cos(theta - target_dir.at<float>(it1.pos())))
                        > direction_consistency_thresh_) ) {
//                    std::cout << "dir=" << target_dir.at<float>(it1.pos()) << ";;;theta=" << theta << "\n";
                    best_match = it1.pos();
                    found_match = true;
                    break;
                }
            }
#else
            if (cos_th < 0) {
                // rotate around origin
                cos_th = -cos_th;
                sin_th = -sin_th;
            }
            // cos(th) is positive by construction
            float l1 = std::min((float)search_line_length_, (cols - 1 - uv0.x) / (cos_th + eps));
            if (sin_th > 0) {
                l1 = std::min(l1, (rows - 1 - uv0.y) / (sin_th + eps));
            } else {
                l1 = std::min(l1, (uv0.y - 1) / (-sin_th + eps));
            }

            cv::Point uv1(uv0.x + l1 * cos_th,
                          uv0.y + l1 * sin_th);
            BresenhamLineIterator it1(uv0.x, uv0.y, uv1.x, uv1.y);
            for (int j = 0; j < it1.size(); ++j, ++it1) {
                if (target.at<uint8_t>(it1.y(), it1.x()) > intensity_threshold_
                    && (!use_dir ||
                        fabs(cos(theta - target_dir.at<float>(it1.y(), it1.x()) ))
                        > direction_consistency_thresh_ ) ) {
                    best_match.x = it1.x();
                    best_match.y = it1.y();
                    found_match = true;
                    break;
                }
            }
#endif

            float len = found_match ? cv::norm(best_match - uv0) : search_line_length_;

#ifdef FEH_USE_OPENCV_LINEITERATOR
            cv::Point uv2{uv0.x-len*cos_th, uv0.y-len*sin_th};
            cv::LineIterator it2(target, uv0, uv2);
            for (int j = 0; j < it2.count; ++j, ++it2) {
                if (*(*it2) > intensity_threshold_
                    && (!use_dir ||
                        fabs(cos(theta - target_dir.at<float>(it2.pos())))
                        > direction_consistency_thresh_)) {
//                    std::cout << "dir=" << target_dir.at<float>(it2.pos()) << ";;;theta=" << theta << "\n";
                    best_match = it2.pos();
                    found_match = true;
                    break;
                }
            }
#else
            // cos(th) positive by construction
            float l2 = std::min(len, uv0.x / (cos_th + eps));
            if (sin_th > 0) {
                l2 = std::min(l2, uv0.y / (sin_th + eps));
            } else {
                l2 = std::min(l2, (rows - uv0.y) / (-sin_th + eps));
            }
            cv::Point uv2{uv0.x - l2 * cos_th, uv0.y - l2 * sin_th};
            BresenhamLineIterator it2(uv0.x, uv0.y, uv2.x, uv2.y);
            for (int j = 0; j < it2.size(); ++j, ++it2) {
                if (target.at<uint8_t>(it2.y(), it2.x()) > intensity_threshold_
                    && (!use_dir ||
                        fabs(cos(theta - target_dir.at<float>(it2.y(), it2.x()) ))
                            > direction_consistency_thresh_ ) ) {
                    best_match.x = it2.x();
                    best_match.y = it2.y();
                    found_match = true;
                    break;
                }
            }
#endif

            if (found_match) {
                match_buffer[i].Set(uv0.x, uv0.y, best_match.x, best_match.y);
                match_buffer[i].Set(edgelist[i]);
            }
        }
    };

    tbb::parallel_for(tbb::blocked_range<size_t>(0, edgelist.size()),
                      kernel,
                      tbb::auto_partitioner());

    matches.clear();
    matches.reserve(edgelist.size());
    for (const auto &match : match_buffer) {
        if (match.IsValid()) {
            matches.push_back(std::move(match));
        }
    }
}


}   // feh

