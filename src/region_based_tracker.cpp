//
// Created by visionlab on 11/2/17.
//

#include "region_based_tracker.h"

// stl
#include <fstream>

// 3rd party
#include "igl/writeOFF.h"
#include "json/json.h"

// own
#include "parallel_kernels.h"
#include "tracker_utils.h"

namespace feh {

namespace tracker {

RegionBasedTracker::RegionBasedTracker():
    timer_("region based tracker"),
    levels_(0),
    inflate_size_(8),
    histogram_size_(32),
    alpha_f_(0.01),
    alpha_b_(0.05),
    hist_code_{"b", "g", "r"},
    constrain_rotation_{false}
{}

void RegionBasedTracker::Initialize(const std::string &config_file,
                                    const std::vector<float> &camera_params,
                                    const MatXf &vertices,
                                    const MatXi &faces) {
    std::string content;
    // folly::readFile(config_file.c_str(), content);
    // config_ = folly::parseJson(folly::json::stripComments(content));
    config_ = LoadJson(config_file);
    // scale levels
    levels_ = config_["levels"].asInt();

    // camera parameters
    if (camera_params.empty()) {
        fx_ = config_["camera"]["fx"].asFloat();
        fy_ = config_["camera"]["fy"].asFloat();
        cx_ = config_["camera"]["cx"].asFloat();
        cy_ = config_["camera"]["cy"].asFloat();
        rows_ = config_["camera"]["rows"].asInt();
        cols_ = config_["camera"]["cols"].asInt();
        fx_ *= cols_;
        fy_ *= rows_;
        cx_ *= cols_;
        cy_ *= rows_;
        cy_ -= 50;
        rows_ = 500;
    } else {
        fx_ = camera_params[0];
        fy_ = camera_params[1];
        cx_ = camera_params[2];
        cy_ = camera_params[3];
        rows_ = (int) camera_params[4];
        cols_ = (int) camera_params[5];
    }



    // near and far plane
    float z_near, z_far;
    z_near = config_["camera"]["z_near"].asFloat();
    z_far = config_["camera"]["z_far"].asFloat();

    for (int i = 0; i < levels_; ++i) {
        RendererPtr renderer =
            std::make_shared<Renderer>(rows_ * powf(0.5, i),
                                       cols_ * powf(0.5, i));
        renderer->SetCamera(z_near, z_far,
                            fx_ * powf(0.5, i),
                            fy_ * powf(0.5, i),
                            cx_ * powf(0.5, i),
                            cy_ * powf(0.5, i));
        renderers_.push_back(renderer);
    }
    roi_.resize(levels_);

    // allocate spaces
    for (int i = 0; i < levels_; ++i) {
        RendererPtr r(renderers_[i]);
        image_pyr_.emplace_back(r->rows(), r->cols(), CV_8UC3);
        depth_.emplace_back(r->rows(), r->cols(), CV_32FC1);
        mask_.emplace_back(r->rows(), r->cols(), CV_8UC1);
        contour_.emplace_back(r->rows(), r->cols(), CV_32FC1);
        distance_.emplace_back(r->rows(), r->cols(), CV_32FC1);
        distance_index_.emplace_back(r->rows(), r->cols(), CV_32SC2);
        signed_distance_.emplace_back(r->rows(), r->cols(), CV_32FC1);
        dsdf_dxp_.emplace_back(r->rows(), r->cols(), CV_32FC2);
        heaviside_.emplace_back(r->rows(), r->cols(), CV_32FC2);
        std::vector<cv::Mat> tmp{cv::Mat{r->rows(), r->cols(), CV_32FC1},
                                 cv::Mat{r->rows(), r->cols(), CV_32FC1}};
        P_.push_back(tmp);
    }

    if (vertices.size() == 0 || faces.size() == 0) {
        // load mesh and setup
        std::tie(vertices_, faces_) = LoadMesh(config_["model"]["path"].asString());
        NormalizeVertices(vertices_);
        if (config_["model"]["scanned"].asBool()) {
            RotateVertices(vertices_, -M_PI / 2.0f);
        } else {
            FlipVertices(vertices_);
        }

        for (auto renderer_ptr : renderers_ ) {
            renderer_ptr->SetMesh(vertices_, faces_);
        }
    } else {
        vertices_ = vertices;
        faces_ = faces;
        for (auto renderer_ptr : renderers_ ) {
            renderer_ptr->SetMesh(vertices_, faces_);
        }
    }

    // setup optimization related member variables
    histogram_size_ = config_["bins_per_channel"].asInt();
//    hist_f_.resize(3);
//    hist_b_.resize(3);
//    for (int i = 0; i < 3; ++i) {
//        hist_f_[i].setZero(histogram_size_);
//        hist_b_[i].setZero(histogram_size_);
//    }
    alpha_f_ = config_["alpha_f"].asDouble();
    alpha_b_ = config_["alpha_b"].asFloat();
    inflate_size_ = config_["inflate_size"].asInt();

    constrain_rotation_ = config_.get("constrain_rotation", false).asBool();
}

void RegionBasedTracker::InitializeTracker(const cv::Mat &image,
                                           const cv::Rect &bbox,
                                           const SE3 &gm_init) {
    cv::Mat edge(renderers_[0]->rows(), renderers_[0]->cols(), CV_8UC1);
    renderers_[0]->RenderEdge(gm_init.matrix(), edge.data);
    cv::Mat display(image.clone());
    cv::rectangle(display, bbox, cv::Scalar(0, 0, 255), 1);
    OverlayMaskOnImage(edge, display, false, kColorGreen);
    cv::imshow("init view", display);

    gm_ = Optimize(image, bbox, gm_init);
}
void RegionBasedTracker::Update(const cv::Mat &image) {
    std::vector<EdgePixel> edgelist;
    renderers_[0]->ComputeEdgePixels(gm_.matrix(), edgelist);
    auto bbox = tracker::RectEnclosedByContour(edgelist,
                                               renderers_[0]->rows(),
                                               renderers_[0]->cols());
    gm_ = Optimize(image, bbox, gm_);
}

SE3 RegionBasedTracker::Optimize(const cv::Mat &image, 
    const cv::Rect &bbox) {
    Vec3f W = GetVectorFromJson<float, 3>(config_, "W0");
    Vec3f T = GetVectorFromJson<float, 3>(config_, "T0");
    SE3 gm(SO3::exp(W), T);
    return Optimize(image, bbox, gm);
}

SE3 RegionBasedTracker::Optimize(const cv::Mat &image,
                                          const cv::Rect &bbox,
                                          const SE3 &gm) {
    // build image pyramid
    image_pyr_[0] = image.clone();
    roi_[0] = bbox;

    for (int i = 1; i < levels_; ++i) {
        cv::Size size(renderers_[i]->cols(), renderers_[i]->rows());
//        std::cout << size << "\n";
//        std::cout << "pyr(i-1) size=" << image_pyr_[i-1].size() << "\n";
//        std::cout << "pyr(i) size=" << image_pyr_[i].size() << "\n";
        cv::pyrDown(image_pyr_[i - 1],
                    image_pyr_[i],
                    size);
        roi_[i].x = roi_[i - 1].x * 0.5;
        roi_[i].y = roi_[i - 1].y * 0.5;
        roi_[i].width = roi_[i - 1].width * 0.5;
        roi_[i].height = roi_[i - 1].height * 0.5;
    }

    SE3 g(gm);
    for (int level = levels_-1; level >= 1; --level) {
        int num_iter = config_["num_iter"].get("level" + std::to_string(level),
                                               10).asInt();
        for (int i = 0; i < num_iter; ++i) {
            std::cout << "@level#" << level << ";;;iter#" << i << "\n";
            bool status = UpdateOneStepAtLevel(level, g);
            if (!status) break;
        }
    }
    cv::destroyAllWindows();

    return g;
}

bool RegionBasedTracker::UpdateOneStepAtLevel(int level, SE3 &g) {
    timer_.Tick("total at level " + std::to_string(level));
    CHECK(level >= 0 && level < levels_) << "level out-of-range";
    // grab local variables
    const cv::Rect &bbox(roi_[level]);
    const cv::Mat &image(image_pyr_[level]);
    cv::Mat &depth(depth_[level]);
    cv::Mat &mask(mask_[level]);
    cv::Mat &contour(contour_[level]);
    cv::Mat &dt(distance_[level]);
    cv::Mat &dt_index(distance_index_[level]);
    cv::Mat &signed_distance(signed_distance_[level]);
    cv::Mat &dsdf_dxp(dsdf_dxp_[level]);
    cv::Mat &heaviside(heaviside_[level]);
    std::vector<cv::Mat> &P(P_[level]);

    RendererPtr renderer(renderers_[level]);

    // render depth
    timer_.Tick("render");
    renderer->RenderDepth(g.matrix(), (float*)depth.data);
    timer_.Tock("render");

    timer_.Tick("binarization");
    tbb::parallel_for(tbb::blocked_range<int>(0, renderer->rows()),
                      BinarizeKernel<float>(depth,
                                            mask,
                                            config_["depth_binarization_threshold"].asFloat()),
                      tbb::auto_partitioner());
    timer_.Tock("binarization");

    // build histogram
    timer_.Tick("build histogram");
    cv::Rect inflated_bbox;
    ComputeColorHistograms(image, mask, inflated_bbox, hist_f_, hist_b_);
    // FIXME: update color histograms with new segmentation but keep optimization constrained to initial bbox
//    ComputeColorHistograms(image, bbox, inflated_bbox, hist_f_, hist_b_);
    timer_.Tock("build histogram");

//    // overwrite inflated_bbox with fixed bbox
//    inflated_bbox = cv::Rect(cv::Point(std::max(bbox.x - inflate_size_, 0),
//                                       std::max(bbox.y - inflate_size_, 0)),
//                             cv::Point(std::min(bbox.x + bbox.width + inflate_size_, image.cols),
//                                       std::min(bbox.y + bbox.height + inflate_size_, image.rows)));


    timer_.Tick("contour extraction");
    tbb::parallel_for(tbb::blocked_range<int>(0, renderer->rows()),
                      EdgeDetectionKernel<float>(depth,
                                                 contour,
                                                 config_["contour_detection_threshold"].asFloat()),
                      tbb::auto_partitioner());
    timer_.Tock("contour extraction");
    CHECK_EQ(contour.type(), CV_32FC1);
    std::cout << "contour detection threshold=" << config_["contour_detection_threshold"].asFloat() << "\n";


    timer_.Tick("distance transformation");
    distance_transformer_(contour, dt, dt_index);
    timer_.Tock("distance transformation");

    timer_.Tick("square root of dt");
    tbb::parallel_for(tbb::blocked_range<int>(0, renderer->rows()),
                      [&dt](const tbb::blocked_range<int> &range) {
                          for (int i = range.begin(); i < range.end(); ++i) {
                              for (int j = 0; j < dt.cols; ++j) {
                                  dt.at<float>(i, j) = sqrt(dt.at<float>(i, j));
                              }
                          }
                      },
                      tbb::auto_partitioner());
    timer_.Tock("square root of dt");

    timer_.Tick("signed distance function");
    tbb::parallel_for(tbb::blocked_range<int>(0, renderer->rows()),
                      SignedDistanceKernel(mask, dt, signed_distance),
                      tbb::auto_partitioner());
    timer_.Tock("signed distance function");

    timer_.Tick("differentiate sdf");
//    tbb::parallel_for(tbb::blocked_range<int>(0, renderer->rows()),
//                      CentralDifferenceKernel(signed_distance, dsdf_dxp),
//                      tbb::auto_partitioner());
    cv::Mat dsdf_dx, dsdf_dy;
    cv::Scharr(signed_distance, dsdf_dx, CV_32F, 1, 0, 3);
    cv::Scharr(signed_distance, dsdf_dy, CV_32F, 0, 1, 3);
//    cv::Sobel(signed_distance, dsdf_dx, CV_32F, 1, 0, 7);
//    cv::Sobel(signed_distance, dsdf_dy, CV_32F, 0, 1, 7);
    cv::merge(std::vector<cv::Mat>{dsdf_dx, dsdf_dy}, dsdf_dxp);
    timer_.Tock("differentiate sdf");

    timer_.Tick("heaviside function");
    float reach = config_["reach_of_smooth_heaviside"].asFloat();
    tbb::parallel_for(tbb::blocked_range<int>(0, renderer->rows()),
                      HeavisideKernel(signed_distance, heaviside, reach),
                      tbb::auto_partitioner());
    timer_.Tock("heaviside function");

    float area_f(0), area_b(0);
    for (int i = inflated_bbox.y; i < inflated_bbox.y + inflated_bbox.height; ++i) {
        for (int j = inflated_bbox.x; j < inflated_bbox.x + inflated_bbox.width; ++j) {
            area_f += heaviside.at<cv::Vec2f>(i, j)[0];
            area_b += (1 - heaviside.at<cv::Vec2f>(i, j)[0]);
        }
    }
    std::cout << "area_f=" << area_f << "/" << inflated_bbox.area() <<"\n";
    std::cout << "area_b=" << area_b << "/" << inflated_bbox.area() <<"\n";

    timer_.Tick("pixelwise posterior");
//    P.setTo(0);
    P[0].setTo(0);
    P[1].setTo(1.0 / area_b);
    tbb::parallel_for(tbb::blocked_range<int>(inflated_bbox.y, inflated_bbox.y + inflated_bbox.height),
                      PixelwisePosteriorKernel(image,
                                               hist_f_,
                                               hist_b_,
                                               area_f,
                                               area_b,
                                               P,
                                               inflated_bbox),
                      tbb::auto_partitioner());
    timer_.Tock("pixelwise posterior");

//    timer_.Tick("evaluate cost");
//    float E(0);
//    for (int i = inflated_bbox.y; i < inflated_bbox.y + inflated_bbox.height; ++i) {
//        for (int j = inflated_bbox.x; j < inflated_bbox.x + inflated_bbox.width; ++j) {
////    for (int i = 0; i < heaviside.rows; ++i) {
////        for (int j = 0; j < heaviside.cols; ++j) {
//            float h = heaviside.at<cv::Vec2f>(i, j)[0];
////            const cv::Vec2f &pfpb = P.at<cv::Vec2f>(i, j);
//            float pf = P[0].at<float>(i, j);
//            float pb = P[1].at<float>(i, j);
//            E += -log(h * pf + (1 - h) * pb);
//        }
//    }
//    timer_.Tock("evaluate cost");
//    std::cout << "cost=" << E << "\n";

    timer_.Tick("compute dxp_dtwist");
    float fx = renderer->fx();
    float fy = renderer->fy();
    float cx = renderer->cx();
    float cy = renderer->cy();
    dxp_dtwist_.clear();
    std::vector<float> pointcloud_buffer;
    for (int i = 0; i < contour.rows; ++i) {
        for (int j = 0; j < contour.cols; ++j) {
            if (contour.at<float>(i, j) == 0) {
                float z = LinearizeDepth(
                    depth.at<float>(i, j),
                    renderer->z_near(),
                    renderer->z_far());

                // y = Proj(Xc)
                // Xc = gcm * Xm
                Vec2f xch((j - cx) / fx, (i - cy) / fy);
                Vec3f Xc(z * xch(0), z * xch(1), z);
                Vec2f y(j, i);
                Eigen::Matrix<float, 2, 6> jac; // dy_d[w, t]

                float z_inv = 1.0f / z;
                float z_inv2 = z_inv * z_inv;

//                Eigen::Matrix<float, 2, 2> dy_dxch;
//                dy_dxch << fx, 0,
//                    0, fy;
//                Mat23f dxch_dXc;
//                dxch_dXc << 1 * z_inv, 0, -Xc(0) * z_inv2,
//                            0, 1 * z_inv, -Xc(1) * z_inv2;
//                Mat23f dy_dXc = dy_dxch * dxch_dXc;

                Mat23f dy_dXc;
                dy_dXc << fx * z_inv, 0, -fx * Xc(0) * z_inv2,
                    0, fy * z_inv, -fy * Xc(1) * z_inv2;

                if (!constrain_rotation_) {
                    // 6 DoF pose estimation
                    Mat3f dXc_dW, dXc_dT;
#define FEH_USE_LINEARIZED_ROTATION
#ifndef FEH_USE_LINEARIZED_ROTATION
                    /// Xc = g*Xm = R*Xm + T, where Xm is the point in model frame, Xc in camera frame
                    //                Vec3f Xm = g.inverse() * Xc;
//                Mat3f R;
//                Mat93 dR_dW;
//                RodriguesFwd(g.so3().log(), R, &dR_dW);
//                dAbVectorized(dR_dW, Xm, dXc_dW);
#else
                    // linearized rotation
                    dXc_dW << 0, Xc(2), -Xc(1),
                        -Xc(2), 0, Xc(0),
                        Xc(1), -Xc(0), 0;
#endif

                    dXc_dT.setIdentity();

                    jac << dy_dXc * dXc_dT, dy_dXc * dXc_dW;
                } else {
                    // rotation is constrained to yaw

                    // FIXME: compute dy_dstate
//                    Vec3f Xm = g.inverse() * Xc;
//                    Mat93 dRinc_da; // where Rinc is the incremental amount added to left of the original R, a is azimuth, i.e., Rnew = Rinc * R, so Rnew * Xm = Rinc * R * Xm = Rinc * (R * Xm)
                    // Rinc = [[cos(a), 0, -sin(a)],
                    //          [0, 1, 0],
                    //          [sin(a), 0, cos(a)]]
                    // dRinc_da = [[-sin(a), 0, -cos(a)],
                    //              [0, 0, 0],
                    //              [cos(a), 0, -sin(a)]]
                    // when a -> 0, sin(a)->0 and cos(a)->1
                    // Thus dRinc_da = [[0, 0, -1],
                    //                  [0, 0, 0],
                    //                  [1, 0, 0]]
//                    dRinc_da.col(0).setZero();
//                    dRinc_da.col(2).setZero();
//                    dRinc_da.col(1) << 0, 0, -1, 0, 0, 0, 1, 0, 0;
//                    dRinc_da.col(1) << 0, 0, 1, 0, 0, 0, -1, 0, 0;
//                    Mat93 dR_da;
//                    AdBVectorized(g.so3().matrix(), dRinc_da, dR_da);
//                    Mat3f dXc_da;
//                    dAbVectorized(dR_da, Xm, dXc_da);
//                    std::cout << "dXc_da=\n" << dXc_da << "\n";
//                    CHECK_EQ(dXc_da.col(0).norm(), 0);
//                    CHECK_EQ(dXc_da.col(2).norm(), 0);


                    Mat3f dXc_dW;
                    dXc_dW << 0, Xc(2), 0,
                            0, 0, 0,
                            0, -Xc(0), 0;

                    jac.block<2, 3>(0, 0) << dy_dXc;
                    jac.block<2, 1>(0, 3) << dy_dXc * dXc_dW.col(1);

                }
                CHECK_EQ(dxp_dtwist_.count(i * contour.cols + j), 0);
                dxp_dtwist_[i * contour.cols + j] = jac;

                pointcloud_buffer.insert(pointcloud_buffer.end(), {Xc(0), Xc(1), Xc(2)});
#ifdef FEH_PWP_DEBUG
                // debug
//                std::cout << z << " ";
                // numeric verification
                Eigen::Matrix<float, 2, 6> num_jac;
                float delta = 1e-4;
                for (int i = 0; i < 3; ++i) {
                    Vec3f Wp = W;
                    Wp(i) += delta;
                    Mat3f Rp;
                    RodriguesFwd(Wp, Rp);
                    Vec3f Xcp = Rp * Xm + T;
                    Vec2f xc(Xcp.head<2>() / Xcp(2));
                    Vec2f xp(xc(0) * fx + cx, xc(1) * fy + cy);
                    CHECK_LE((xp-y).norm() / (xp.norm() + y.norm()), 1e-3);
                    num_jac.block<2, 1>(0, i) = (xp - y) / delta;
                }
                Mat3f Rp;
                RodriguesFwd(W, Rp);
                for (int i = 0; i < 3; ++i) {
                    Vec3f Tp = T;
                    Tp(i) += delta;
                    Vec3f Xcp = Rp * Xm + Tp;
                    Vec2f xc(Xcp.head<2>() / Xcp(2));
                    Vec2f xp(xc(0) * fx + cx, xc(1) * fy + cy);
                    CHECK_LE((xp-y).norm() / (xp.norm() + y.norm()), 1e-3);
                    num_jac.block<2, 1>(0, 3 + i) = (xp - y) / delta;
                }

                CHECK_LE((num_jac - jac).norm() / (num_jac.norm() + jac.norm()), 1e-2);
#endif
            }
        }
    }
    timer_.Tock("compute dxp_dtwist");
    std::cout << "#active contour points=" << dxp_dtwist_.size() << "\n";

    std::cout << "\n";
    if (config_["dump_pointcloud"].asBool()){
        // convert pointcloud buffer to vertex matrix
        Eigen::MatrixXf V =
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                (&pointcloud_buffer[0], pointcloud_buffer.size()/3, 3);

        // face matrix
        Eigen::MatrixXi F;

        // setup color
        std::vector<uint8_t> pointcloud_color;
        pointcloud_color.reserve(pointcloud_buffer.size());
        for (int i = 0; i < pointcloud_buffer.size() / 3; ++i) {
            pointcloud_color.insert(pointcloud_color.end(), {255, 0, 0});
        }
        // convert to color matrix
        Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> C =
            Eigen::Map<Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
                (&pointcloud_color[0], pointcloud_color.size()/3, 3);
        // write out points on the boundary with color
        // igl::writeOFF("boundary_pointcloud.off", V, F, CC);

        // apply gcm to model vertices
        Eigen::MatrixXf model_vertices = vertices_ * g.so3().inv().matrix();
        model_vertices.rowwise() += g.translation().transpose();
        // write out
        igl::writeOFF("model_pointcloud.off", model_vertices, F);
    }

    timer_.Tick("construct linear system");
    Eigen::Matrix<float, 6, 6> JtJ;
    Eigen::Matrix<float, 6, 1> Jt;
    std::vector<float> J_buffer, r_buffer;
    std::vector<cv::Mat> J_debug(6, cv::Mat{contour.rows, contour.cols, CV_32FC1});
    JtJ.setZero();
    Jt.setZero();
    for (int i = 0; i < 6; ++i) J_debug[i].setTo(0);

    cv::Mat active_pixels(contour.rows, contour.cols, CV_8UC1);
    active_pixels.setTo(255);
//    for (int i = inflated_bbox.y; i < inflated_bbox.y + inflated_bbox.height; ++i) {
//        for (int j = inflated_bbox.x; j < inflated_bbox.x + inflated_bbox.width; ++j) {
//            if (dt.at<float>(i, j) > 10) continue;
    for (int i = 0; i < contour.rows; ++i) {
        for (int j = 0; j < contour.cols; ++j) {
            if (dt.at<float>(i, j) > inflate_size_) continue;

            const float heaviside_value = heaviside.at<cv::Vec2f>(i, j)(0);   // h value and dh_dsdf
            const float dh_dsdf_value = heaviside.at<cv::Vec2f>(i, j)(1);
            const cv::Vec2f &dsdf_dxp_value = dsdf_dxp.at<cv::Vec2f>(i, j);
            Vec2f dh_dxp(dh_dsdf_value * dsdf_dxp_value(0),
                         dh_dsdf_value * dsdf_dxp_value(1));
            int dxp_dtwist_index;
            // only operate on contour pixels
//            if (contour.at<float>(i, j) > 0) continue;
//            if (mask.at<uint8_t>(i, j) == 0) {
//                // distinguish foreground and background
//                dxp_dtwist_index = i * contour.cols + j;
//            } else
            const cv::Vec2i &closest_edge_pixel = dt_index.at<cv::Vec2i>(i, j); // (x, y)
            CHECK(contour.at<float>(closest_edge_pixel(1), closest_edge_pixel(0)) != kBigNumber);
            dxp_dtwist_index = closest_edge_pixel(1) * contour.cols + closest_edge_pixel(0);

            const Eigen::Matrix<float, 2, 6> &dxp_dtwist = dxp_dtwist_.at(dxp_dtwist_index);
            Eigen::Matrix<float, 1, 6> dh_dwt(dh_dxp.transpose() * dxp_dtwist);
//            std::cout << dh_dwt << "\n";
            float Pf = P[0].at<float>(i, j);
            float Pb = P[1].at<float>(i, j);
            float dP_dh = (Pf - Pb)
                          / (heaviside_value * (Pf - Pb) + Pb);
            Eigen::Matrix<float, 1, 6> this_j = -dP_dh * dh_dwt;
            for (int k = 0; k < 6; ++k) J_buffer.push_back(this_j(k));

            float r = -sqrt(-log(heaviside_value * Pf + (1 - heaviside_value) * Pb));
//            float r = -log(heaviside_value * Pf + (1 - heaviside_value) * Pb);
            r_buffer.push_back(r);

            Jt += this_j.transpose() * r;
            JtJ += this_j.transpose() * this_j;

            // debug
            for (int k = 0; k < 6; ++k) J_debug[k].at<float>(i, j) = dh_dwt(k);
//            cv::line(active_pixels,
//                     cv::Point(j, i),
//                     cv::Point(closest_edge_pixel(0), closest_edge_pixel(1)),
//                     cv::Scalar(0));
        }
    }
//    Eigen::MatrixXf J = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
//        (&J_buffer[0], J_buffer.size()/6, 6);
//    Jt = J.transpose() * Eigen::Map<Eigen::VectorXf>(&r_buffer[0], r_buffer.size(), 1);
//    JtJ = J.transpose() * J;

//    // add delta to current state
    Eigen::Matrix<float, 6, 6> I;
    I.setIdentity();
    float translation_damping_factor = config_.get("translation_damping_factor", 0.0).asFloat();
    float rotation_damping_factor = config_.get("rotation_damping_factor", 0.0).asFloat();
    I.block<3, 3>(0, 0) *= translation_damping_factor;
    I.block<3, 3>(3, 3) *= rotation_damping_factor;
    Jt *= config_.get("residual_scaling", 1.0).asFloat();

    if (!constrain_rotation_) {
        Eigen::Matrix<float, 6, 1> delta = -(JtJ + I).llt().solve(Jt);
        std::cout << "delta=" << delta.transpose() << "\n";
        std::cout << "g=" << g.matrix3x4() << "\n";
        // FIXME: exponential map does not work now with my own se3 type
        // g = SE3::exp(delta) * g;
    } else {
        Eigen::Matrix<float, 4, 1> delta
            = -(JtJ + I).block<4, 4>(0, 0).llt().solve(Jt.head<4>());
        std::cout << "delta=" << delta.transpose() << "\n";
        std::cout << "g=" << g.matrix3x4() << "\n";
//        g.setRotationMatrix(g.so3().matrix() * Eigen::AngleAxisf(delta(3), Vec3f::UnitY()));
//        g.translation() += delta.head<3>();

        Eigen::Matrix<float, 6, 1> full_delta;
        full_delta.setZero();
        full_delta.head<3>() = delta.head<3>();
        full_delta(4) = delta(3);

        // FIXME: exponential map does not work now with my own se3 type
        // g = SE3::exp(full_delta) * g;

    }
    timer_.Tock("construct linear system");

    timer_.Tock("total at level " + std::to_string(level));

    if (config_["tracker_view"].asBool()) {
        // visualization at the original scale
        cv::Mat edge(renderers_[0]->rows(), renderers_[0]->cols(), CV_8UC1);
        renderers_[0]->RenderEdge(g.matrix(), edge.data);
//        renderers_[0]->RenderWireframe(g.matrix(), edge.data);
        display_ = image_pyr_[0].clone();
//        cv::rectangle(display, inflated_bbox, cv::Scalar(255, 0, 0), 1);
//        cv::rectangle(display, roi_[0], cv::Scalar(0, 0, 255), 1);
        OverlayMaskOnImage(edge, display_, false, kColorGreen);
    }

    if (config_["dump_mats"].asBool()) {

        std::vector<cv::Mat> hdh(2);
        cv::split(heaviside, hdh);

        std::vector<cv::Mat> pfpb(2);

        std::vector<cv::Mat> dsdf_dxy(2);
        cv::split(dsdf_dxp, dsdf_dxy);

        // check signed distance field
        for (int i = 0; i < dt.rows; ++i) {
            for (int j = 0; j < dt.cols; ++j) {
                if (mask.at<uint8_t>(i, j)) {
                    CHECK_EQ(dt.at<float>(i, j), signed_distance.at<float>(i, j));
                } else {
                    CHECK_EQ(dt.at<float>(i, j), -signed_distance.at<float>(i, j));
                }
            }
        }

        if (config_["visualize"].asBool()) {
            cv::imshow("depth", depth);
            cv::imshow("mask", mask);
            cv::Mat contour_display = DistanceTransform::BuildView(contour);
            cv::imshow("contour", contour_display);
            cv::Mat dt_display = DistanceTransform::BuildView(dt);
            cv::imshow("distance transform", dt_display);
            // check distance transformation indices
            int counter(0);
            cv::Mat index_debug(contour_display.clone());
            for (int i = 0; i < contour.rows; i += 5) {
                for (int j = 0; j < contour.cols; j += 5) {
                    CHECK_LE(dt.at<float>(i, j), powf(contour.rows, 2) + powf(contour.cols, 2));
                    if (contour.at<float>(i, j) == 0) {
                        CHECK_EQ(dt_index.at<cv::Vec2i>(i, j)(0), j);
                        CHECK_EQ(dt_index.at<cv::Vec2i>(i, j)(1), i);
                        ++counter;
                    } else {
                        cv::line(index_debug,
                                 cv::Point(dt_index.at<cv::Vec2i>(i, j)(0),
                                           dt_index.at<cv::Vec2i>(i, j)(1)),
                                 cv::Point(j, i), cv::Scalar(0), 1);
                    }
                }
            }
            cv::imshow("distance transform index", index_debug);

            cv::Mat sdf_display = DistanceTransform::BuildView(signed_distance);
            cv::imshow("signed distance field", sdf_display);

            cv::Mat heaviside_display = DistanceTransform::BuildView(hdh[0]);
            cv::imshow("heaviside field", heaviside_display);
            cv::Mat dh_dsdf_display = DistanceTransform::BuildView(hdh[1]);
            cv::imshow("dh_dsdf", dh_dsdf_display);
            cv::Mat pf_display = DistanceTransform::BuildView(P[0]);
            cv::imshow("Pf", pf_display);
            cv::Mat pb_display = DistanceTransform::BuildView(P[1]);
            cv::imshow("Pb", pb_display);
            cv::Mat dsdf_dx_display = DistanceTransform::BuildView(dsdf_dxy[0]);
            cv::Mat dsdf_dy_display = DistanceTransform::BuildView(dsdf_dxy[1]);
            cv::imshow("dsdf_dx", dsdf_dx_display);
            cv::imshow("dsdf_dy", dsdf_dy_display);
        }
    }

    char ckey = cv::waitKey(config_["wait_time"].asInt());
    if (ckey == 'q') return false;
    std::cout << timer_;
    return true;
}

void RegionBasedTracker::ComputeColorHistograms(
    const cv::Mat &image,
    const cv::Mat &mask,
    cv::Rect &bbox,
    std::vector<VecXf> &histf,
    std::vector<VecXf> &histb) const {

    std::vector<VecXf> histf_old(histf);
    std::vector<VecXf> histb_old(histb);

    feh::tracker::ComputeColorHistograms(image,
                                         mask,
                                         bbox,
                                         histf, histb,
                                         histogram_size_,
                                         inflate_size_);
    // update the histograms
    if (histb_old.empty() || histf_old.empty()) {
//        histf = this_histf;
//        histb = this_histb;
    } else {
        for (int k = 0; k < 3; ++k) {
            histf[k] = (1 - alpha_f_) * histf_old[k] + alpha_f_ * histf[k];
            histb[k] = (1 - alpha_b_) * histb_old[k] + alpha_b_ * histb[k];
        }
    }



#ifdef FEH_PWP_DEBUG
    std::cout << TermColor::bold << TermColor::white << "foreground color histogram" << TermColor::endl;
    std::cout << TermColor::blue << histf[0].transpose() << TermColor::endl;
    std::cout << TermColor::green << histf[1].transpose() << TermColor::endl;
    std::cout << TermColor::red << histf[2].transpose() << TermColor::endl;

    std::cout << TermColor::bold << TermColor::white << "background color histogram" << TermColor::endl;
    std::cout << TermColor::blue << histb[0].transpose() << TermColor::endl;
    std::cout << TermColor::green << histb[1].transpose() << TermColor::endl;
    std::cout << TermColor::red << histb[2].transpose() << TermColor::endl;

//    for (int i = 0; i <3; ++i) {
//        plt::clf();
//        plt::plot(std::vector<float>{histf[i].data(), histf[i].data()+histf[i].size()}, hist_code_[i] + "o");
//        plt::plot(std::vector<float>{histb[i].data(), histb[i].data()+histb[i].size()}, hist_code_[i] + "+");
//        plt::save("histogram_channel_" + hist_code_[i] + ".png");
//    }
#endif


}

void RegionBasedTracker::ComputeColorHistograms(
    const cv::Mat &image,
    const cv::Rect &bbox,
    cv::Rect &inflated_bbox,
    std::vector<VecXf> &histf,
    std::vector<VecXf> &histb) const {

    std::vector<VecXf> histf_old(histf);
    std::vector<VecXf> histb_old(histb);

    feh::tracker::ComputeColorHistograms(image,
                                         bbox,
                                         inflated_bbox,
                                         histf, histb,
                                         histogram_size_,
                                         inflate_size_);

    // update the histograms
    if (histb_old.empty() || histf_old.empty()) {
//        histf = this_histf;
//        histb = this_histb;
    } else {
        for (int k = 0; k < 3; ++k) {
            histf[k] = (1 - alpha_f_) * histf_old[k] + alpha_f_ * histf[k];
            histb[k] = (1 - alpha_b_) * histb_old[k] + alpha_b_ * histb[k];
        }
    }

}

}   // namespace tracker
}   // namespace feh
