//
// Created by feixh on 11/15/17.
//
#include "dataloaders.h"

// 3rd party
#include "json/json.h"

// own
#include "tracker_utils.h"
#include "opencv2/imgproc.hpp"

namespace feh {

LinemodDatasetLoader::LinemodDatasetLoader(const std::string &dataroot):
    dataroot_(dataroot) {
    std::tie(vertices_, faces_) = LoadMesh(dataroot + "/mesh.ply");
    tracker::ScaleVertices(vertices_, 1e-3);

    // transformation to register oldmesh
    std::string transform_file = dataroot_ + "/transform.dat";
    std::ifstream ifs(transform_file, std::ios::in);
    CHECK(ifs.is_open()) << "failed to open transform.dat at " << transform_file;
    float tmp;
    ifs >> tmp;
    Mat4f transform_data;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            ifs >> tmp >> tmp;
            transform_data(i, j) = tmp;
        }
    }
    transform_data(3, 3) = 1.0;
    ifs.close();
    transform_ = Sophus::SE3f(transform_data);
    std::cout << "transform=\n" << transform_.matrix() << "\n";

    // read in jpg files
    Glob(dataroot_ + "/data", "jpg", "color", jpg_files_);
    Glob(dataroot_ + "/data", "tra", "tra", trans_files_);
    Glob(dataroot_ + "/data", "rot", "rot", rot_files_);
    CHECK_EQ(jpg_files_.size(), trans_files_.size());
    CHECK_EQ(jpg_files_.size(), rot_files_.size());
    size_ = jpg_files_.size();
}

bool LinemodDatasetLoader::Grab(int i,
                                cv::Mat &image,
                                Sophus::SE3f &gm) {
    if (i >= size_ || i < 0) return false;

    image = cv::imread(jpg_files_[i]);

    // read in translation
    std::ifstream ifs(trans_files_[i], std::ios::in);
    CHECK(ifs.is_open());
    int tmp;
    ifs >> tmp >> tmp;
    Vec3f T;
    ifs >> T(0) >> T(1) >> T(2);
    T *= 0.01;  // cm -> meter
    ifs.close();

    // read in rotation
    ifs.open(rot_files_[i], std::ios::in);
    CHECK(ifs.is_open());
    ifs >> tmp >> tmp;
    Mat3f R;
    for (int i =0 ; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ifs >> R(i, j);
        }
    }
    ifs.close();

    // fill in homogeneous matrix
    Mat4f RT;
    RT.block<3, 3>(0, 0) = R;
    RT.block<3, 1>(0, 3) = T;
    RT(3, 3) = 1.0;

    gm = Sophus::SE3f(RT);

    return true;
}

RigidPoseDatasetLoader::RigidPoseDatasetLoader(
    const std::string &dataroot,
    const std::string &dataset,
    int tag):
    dataroot_(dataroot),
    dataset_(dataset),
    index_(0){

    // strip leading and tailing "/"
    for (auto it = dataset_.begin(); it != dataset_.end(); ) {
        if (*it == '/') {
            it = dataset_.erase(it);
        } else ++it;
    }

    while (dataset_.back() == '/') {
        dataset_.pop_back();
    }

    // parse tag
    std::string video_path(dataroot_ + "/" + dataset_ + "/");
    if (tag & as_integer(Tag::Left)) {
        if (tag & as_integer(Tag::NoiseFree)) {
            video_path += dataset_ + "_noise_free_L.mp4";
        } else if (tag & as_integer(Tag::Noisy)) {
            video_path += dataset_ + "_noisy_L.mp4";
        } else if (tag & as_integer(Tag::Occluded)) {
            video_path += dataset_ + "_occluded_L.mp4";
        } else {
            LOG(FATAL) << "Invalid tag L";
        }
    } else if (tag & as_integer(Tag::Right)) {
        if (tag & as_integer(Tag::NoiseFree)) {
            video_path += dataset_ + "_noise_free_R.mp4";
        } else if (tag & as_integer(Tag::Noisy)) {
            video_path += dataset_ + "_noisy_R.mp4";
        } else if (tag & as_integer(Tag::Occluded)) {
            video_path += dataset_ + "_occluded_R.mp4";
        } else {
            LOG(FATAL) << "Invalid tag R";
        }
    } else {
        LOG(FATAL) << "Invalid tag";
    }
    LOG(INFO) << "loading video at " << video_path;

    capture_.open(video_path);
    CHECK(capture_.isOpened()) << "failed to open video @ " << video_path;

    // load model
    std::string model_path(dataroot_ + "/models/" + dataset_ + "/" + dataset_ + ".obj");
    LoadMesh(model_path, vertices_, faces_);
    tracker::ScaleVertices(vertices_, 1e-3);   // mm -> meters
    // flip z
    vertices_.col(2) *= -1;
    LOG(INFO) << vertices_.size() / 3 << " vertices loaded";
    LOG(INFO) << faces_.size() / 3 << " faces loaded";

    // load ground truth poses
    std::string pose_path(dataroot_  + "/code/ground_truth.txt");
    std::fstream ifs(pose_path, std::ios::in);
    CHECK(ifs.is_open()) << "failed to open file @ " << pose_path;
    // skip first two lines
    std::string ignore;
    std::getline(ifs, ignore);
    std::getline(ifs, ignore);

    for (Eigen::Matrix<float, 6, 1> v; ifs >> v(0) >> v(1) >> v(2) >> v(3) >> v(4) >> v(5); ) {
        std::cout << v.transpose() << "\n";
        v.head<3>() *= 1e-3;    // mm -> meters
        g_.push_back(v);
    }
    ifs.close();

}

bool RigidPoseDatasetLoader::Grab(cv::Mat &image, Sophus::SE3f &pose) {
    capture_ >> image;
    if (image.empty()) return false;
    if (index_ >= g_.size()) return false;
    pose = GetPose(index_);
    ++index_;
    return true;
}

Sophus::SE3f RigidPoseDatasetLoader::GetPose(int i) const {
//    auto R = Sophus::SO3f::exp(g_[i].tail<3>());
//    R *= Sophus::SO3f((Eigen::Matrix3f ()<< -1, 0, 0,
//                      0, -1, 0,
//                      0, 0, 1).finished());
    Eigen::Matrix3f R = Eigen::AngleAxisf(g_[i].tail<3>().norm(), g_[i].tail<3>()).toRotationMatrix();
    return Sophus::SE3f(Sophus::SO3f::fitToSO3(R), g_[i].head<3>());
}

VlslamDatasetLoader::VlslamDatasetLoader(const std::string &dataroot):
dataroot_(dataroot) {

    std::ifstream in_file(dataroot_ + "/dataset");
    CHECK(in_file.is_open()) << "failed to open dataset";

    dataset_.ParseFromIstream(&in_file);
    in_file.close();

    if (!Glob(dataroot_, ".png", png_files_)) {
        LOG(FATAL) << "FATAL::failed to read png file list @" << dataroot_;
    }

//    for (int i = 0; i < png_files_.size(); ++i) {
//        std::cout << png_files_[i] << "\n";
//    }


    if (!Glob(dataroot_, ".edge", edge_files_)) {
        LOG(FATAL) << "FATAL::failed to read edge map list @" << dataroot_;
    }

    if (!Glob(dataroot_, ".bbox", bbox_files_)) {
        LOG(FATAL) << "FATAL::failed to read bounding box lisst @" << dataroot_;
    }

    CHECK_EQ(png_files_.size(), edge_files_.size());
    CHECK_EQ(png_files_.size(), bbox_files_.size());
    size_ = png_files_.size();
}


bool VlslamDatasetLoader::Grab(int i,
                               cv::Mat &image,
                               cv::Mat &edgemap,
                               vlslam_pb::BoundingBoxList &bboxlist,
                               Sophus::SE3f &gwc,
                               Sophus::SO3f &Rg,
                               std::string &fullpath) {
    fullpath = png_files_[i];
    return Grab(i, image, edgemap, bboxlist, gwc, Rg);
}

bool VlslamDatasetLoader::Grab(int i,
                               cv::Mat &image,
                               cv::Mat &edgemap,
                               vlslam_pb::BoundingBoxList &bboxlist,
                               Sophus::SE3f &gwc,
                               Sophus::SO3f &Rg) {

    if (i >= size_ || i < 0) return false;
//    std::cout << i << "\n";

    vlslam_pb::Packet *packet_ptr(dataset_.mutable_packets(i));
    gwc = Sophus::SE3f(SE3FromArray(packet_ptr->mutable_gwc()->mutable_data()));

    // gravity alignment rotation
    Vec3f Wg(packet_ptr->wg(0), packet_ptr->wg(1), 0);
    Rg = Sophus::SO3f::exp(Wg);

    std::string png_file = png_files_[i];
    std::string edge_file = edge_files_[i];
    std::string bbox_file = bbox_files_[i];

    // read image
    image = cv::imread(png_file);
    CHECK(!image.empty()) << "empty image: " << png_file;

    // read edgemap
    if (!LoadEdgeMap(edge_file, edgemap)) {
        LOG(FATAL) << "failed to load edge map @ " << edge_file;
    }

    // read bounding box
    std::ifstream in_file(bbox_file, std::ios::in);
    CHECK(in_file.is_open()) << "FATAL::failed to open bbox file @ " << bbox_file;
    bboxlist.ParseFromIstream(&in_file);
    in_file.close();
    return true;
}

std::unordered_map<int64_t, std::array<double, 6>> VlslamDatasetLoader::GrabPointCloud(int i,
                                                                                  const cv::Mat &img) {
    std::unordered_map<int64_t, std::array<double, 6>> out;
    vlslam_pb::Packet *packet_ptr = dataset_.mutable_packets(i);
    for (auto f : packet_ptr->features()) {
        if (f.status() == vlslam_pb::Feature_Status_INSTATE
            || f.status() == vlslam_pb::Feature_Status_GOODDROP) {
            auto color = img.at<cv::Vec3b>(int(f.xp(1)), int(f.xp(0)));
            if (out.count(f.id())) {
                color[0] += out.at(f.id())[3];
                color[1] += out.at(f.id())[4];
                color[2] += out.at(f.id())[5];
                color[0] >>= 1;
                color[1] >>= 1;
                color[2] >>= 1;
                out[f.id()] = {f.xw(0), f.xw(1), f.xw(2),
                               static_cast<double>(color[0]),
                               static_cast<double>(color[1]),
                               static_cast<double>(color[2])};
            } else {
                out[f.id()] = {f.xw(0), f.xw(1), f.xw(2),
                               static_cast<double>(color[0]),
                               static_cast<double>(color[1]),
                               static_cast<double>(color[2])};
            }
        }
    }
    return out;
};

ICLDatasetLoader::ICLDatasetLoader(const std::string &dataroot) {
    dataroot_ = dataroot;
    // load camera pose
    std::ifstream fid(dataroot_ + "/traj", std::ios::in);
    CHECK(fid.is_open()) << "failed to open pose file";
    float tmp[12];
    for (;;) {
        Sophus::SE3f pose;
        for (int i = 0; i < 12; ++i) if (!(fid >> tmp[i])) break;
        pose.setRotationMatrix((Mat3f() << tmp[0], tmp[1], tmp[2],
            tmp[4], tmp[5], tmp[6],
            tmp[8], tmp[9], tmp[10]).finished());
        pose.translation() << tmp[3], tmp[7], tmp[11];
        poses_.push_back(pose);
//        std::cout << "pose=\n" << pose.matrix3x4() << "\n";
        if (fid.eof()) break;
    }
    fid.close();

    if (!Glob(dataroot_, ".png", png_files_)) {
        LOG(FATAL) << "FATAL::failed to read png file list @" << dataroot_;
    }
    // remove leading and trailing item
    png_files_.pop_back();
    png_files_.erase(png_files_.begin());

    if (!Glob(dataroot_, ".edge", edge_files_)) {
        LOG(FATAL) << "FATAL::failed to read edge map list @" << dataroot_;
    }
    edge_files_.pop_back();
    edge_files_.erase(edge_files_.begin());

    if (!Glob(dataroot_, ".bbox", bbox_files_)) {
        LOG(FATAL) << "FATAL::failed to read bounding box lisst @" << dataroot_;
    }
    bbox_files_.pop_back();
    bbox_files_.erase(bbox_files_.begin());

    CHECK_EQ(png_files_.size(), edge_files_.size());
    CHECK_EQ(png_files_.size(), bbox_files_.size());
//    CHECK_EQ(png_files_.size(), poses_.size());
    poses_.resize(png_files_.size());   // HACK: STUPID FILE IO
    size_ = png_files_.size();
}


bool ICLDatasetLoader::Grab(int i,
                            cv::Mat &image,
                            cv::Mat &edgemap,
                            vlslam_pb::BoundingBoxList &bboxlist,
                            Sophus::SE3f &gwc,
                            Sophus::SO3f &Rg) {

    if (i >= size_ || i < 0) return false;
    std::cout << i << "/" << size_ << "\n";


    gwc = poses_[i];
    // world frame is already gravity aligned thanks to the nice ICL-NUIM dataset
    Rg.setQuaternion(Eigen::Quaternion<float>::Identity());

    std::string png_file = png_files_[i];
    std::string edge_file = edge_files_[i];
    std::string bbox_file = bbox_files_[i];

    // read image
    image = cv::imread(png_file);
    CHECK(!image.empty()) << "empty image: " << png_file;

    // read edgemap
    if (!LoadEdgeMap(edge_file, edgemap)) {
        LOG(FATAL) << "failed to load edge map @ " << edge_file;
    }

    // read bounding box
    std::ifstream in_file(bbox_file, std::ios::in);
    CHECK(in_file.is_open()) << "FATAL::failed to open bbox file @ " << bbox_file;
    bboxlist.ParseFromIstream(&in_file);
    in_file.close();
    return true;
}

bool ICLDatasetLoader::Grab(int i,
                               cv::Mat &image,
                               cv::Mat &edgemap,
                               vlslam_pb::BoundingBoxList &bboxlist,
                               Sophus::SE3f &gwc,
                               Sophus::SO3f &Rg,
                               std::string &fullpath) {
    fullpath = png_files_[i];
    return Grab(i, image, edgemap, bboxlist, gwc, Rg);
}

SceneNNDatasetLoader::SceneNNDatasetLoader(const std::string &dataroot) {
    dataroot_ = dataroot;
    // load camera pose
    std::ifstream fid(dataroot_ + "/trajectory.log", std::ios::in);
    CHECK(fid.is_open()) << "failed to open pose file";

    try {
        std::string contents;
        // folly::readFile((dataroot_+"/skip.json").c_str(), contents);
        // folly::dynamic skip_js = folly::parseJson(folly::json::stripComments(contents));
        std::ifstream in{dataroot_+"/skip.json", std::ios::in};
        Json::Reader reader;
        Json::Value skip_js;
        reader.parse(in, skip_js);
        skip_head_ = skip_js.get("head", 0).asInt();
        until_last_ = skip_js.get("last", -1).asInt();
    } catch (...) {
        skip_head_ = 0;
        until_last_ = -1;
    }



    float tmp[16];
    for (;;) {
        Sophus::SE3f pose;
        int x;
        fid >> x; fid >> x; fid >> x;
        for (int i = 0; i < 16; ++i) if (!(fid >> tmp[i])) break;
        pose.so3() = Sophus::SO3f::fitToSO3(
            (Mat3f() << tmp[0], tmp[1], tmp[2],
                tmp[4], tmp[5], tmp[6],
                tmp[8], tmp[9], tmp[10]).finished());
        pose.translation() << tmp[3], tmp[7], tmp[11];
        poses_.push_back(pose);
//        std::cout << "pose=\n" << pose.matrix3x4() << "\n";
        if (fid.eof()) break;
    }
    fid.close();
    size_ = std::min<int>(std::numeric_limits<int>::max(), poses_.size());

    if (!Glob(dataroot_+"/image/", ".png", png_files_)) {
        LOG(FATAL) << "FATAL::failed to read png file list @" << dataroot_;
    }
    // remove leading and trailing item
    size_ = std::min<int>(png_files_.size(), size_);

    if (!Glob(dataroot_+"/image/", ".edge", edge_files_)) {
        LOG(FATAL) << "FATAL::failed to read edge map list @" << dataroot_;
    }
    size_ = std::min<int>(edge_files_.size(), size_);

    if (!Glob(dataroot_+"/image/", ".bbox", bbox_files_)) {
        LOG(FATAL) << "FATAL::failed to read bounding box lisst @" << dataroot_;
    }
    size_ = std::min<int>(bbox_files_.size(), size_);
}


bool SceneNNDatasetLoader::Grab(int i,
                            cv::Mat &image,
                            cv::Mat &edgemap,
                            vlslam_pb::BoundingBoxList &bboxlist,
                            Sophus::SE3f &gwc,
                            Sophus::SO3f &Rg) {
    i += skip_head_;

    if (i >= size_ || i < 0) return false;
    std::cout << i << "/" << size_ << "\n";

    if (until_last_ != -1 && i > until_last_) return false;


    gwc = poses_[i];
    // FIXME: MAKE UP GRAVITY
    Rg.setQuaternion(Eigen::Quaternion<float>::Identity());

    std::string png_file = png_files_[i];
    std::string edge_file = edge_files_[i];
    std::string bbox_file = bbox_files_[i];

    // read image
    image = cv::imread(png_file);
    CHECK(!image.empty()) << "empty image: " << png_file;

    // read edgemap
    if (!LoadEdgeMap(edge_file, edgemap)) {
        LOG(FATAL) << "failed to load edge map @ " << edge_file;
    }

    // read bounding box
    std::ifstream in_file(bbox_file, std::ios::in);
    CHECK(in_file.is_open()) << "FATAL::failed to open bbox file @ " << bbox_file;
    bboxlist.ParseFromIstream(&in_file);
    in_file.close();
    return true;
}

bool SceneNNDatasetLoader::Grab(int i,
                            cv::Mat &image,
                            cv::Mat &edgemap,
                            vlslam_pb::BoundingBoxList &bboxlist,
                            Sophus::SE3f &gwc,
                            Sophus::SO3f &Rg,
                            std::string &fullpath) {
    fullpath = png_files_[i+skip_head_];
    return Grab(i, image, edgemap, bboxlist, gwc, Rg);
}

KittiDatasetLoader::KittiDatasetLoader(const std::string &dataroot) {
    dataroot_ = dataroot;
    Glob(dataroot_, "png", png_files_);
    Glob(dataroot_, "edge", edge_files_);
    Glob(dataroot_, "bbox", bbox_files_);
    std::vector<std::string> pose_files;
    Glob(dataroot_, "txt", pose_files);
    CHECK_EQ(png_files_.size(), edge_files_.size());
    CHECK_EQ(png_files_.size(), bbox_files_.size());
    CHECK_EQ(png_files_.size(), pose_files.size());
    size_ = png_files_.size();

    Mat3f swapaxis;
    swapaxis << 1.0, 0.0, 0.0,
        0.0, 0.0, -1.0,
        0.0, 1.0, 0.0;

    // now load poses
    for (int i = 0; i < pose_files.size(); ++i) {
        // debug
        std::cout << "loading " << i << "/" << size_ << " ... " << pose_files[i] << "\n";
        std::ifstream fid(pose_files[i], std::ios::in);
        CHECK(fid.is_open()) << "failed to open pose file " << pose_files[i];
        float tmp[12];
        for (int k = 0; k < 12; ++k) fid >> tmp[k];
        Sophus::SE3f pose;
        pose.so3() = Sophus::SO3f::fitToSO3(
            swapaxis * (Mat3f() << tmp[0], tmp[1], tmp[2],
                tmp[4], tmp[5], tmp[6],
                tmp[8], tmp[9], tmp[10]).finished());
        pose.translation() << swapaxis * (Vec3f() << tmp[3], tmp[7], tmp[11]).finished();
        poses_.push_back(pose);
        fid.close();
    }
}

bool KittiDatasetLoader::Grab(int i,
                              cv::Mat &image,
                              cv::Mat &edgemap,
                              vlslam_pb::BoundingBoxList &bboxlist,
                              Sophus::SE3f &gwc,
                              Sophus::SO3f &Rg) {
    gwc = poses_[i];
    // FIXME: MAKE UP GRAVITY
    Rg.setQuaternion(Eigen::Quaternion<float>::Identity());

    std::string png_file = png_files_[i];
    std::string edge_file = edge_files_[i];
    std::string bbox_file = bbox_files_[i];

    // read image
    image = cv::imread(png_file);
    cv::resize(image, image, cv::Size(1240, 376));
    CHECK(!image.empty()) << "empty image: " << png_file;

    // read edgemap
    if (!LoadEdgeMap(edge_file, edgemap)) {
        LOG(FATAL) << "failed to load edge map @ " << edge_file;
    }
    cv::resize(edgemap, edgemap, cv::Size(1240, 376));

    // read bounding box
    std::ifstream in_file(bbox_file, std::ios::in);
    CHECK(in_file.is_open()) << "FATAL::failed to open bbox file @ " << bbox_file;
    bboxlist.ParseFromIstream(&in_file);
    in_file.close();
    return true;
}


bool KittiDatasetLoader::Grab(int i,
                              cv::Mat &image,
                              cv::Mat &edgemap,
                              vlslam_pb::BoundingBoxList &bboxlist,
                              Sophus::SE3f &gwc,
                              Sophus::SO3f &Rg,
                              std::string &fullpath) {
    fullpath = png_files_[i];
    return Grab(i, image, edgemap, bboxlist, gwc, Rg);
}



}   // namespace feh
