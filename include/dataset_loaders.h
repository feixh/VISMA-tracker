//
// Created by feixh on 11/15/17.
//
#pragma once

// stl
#include <vector>
#include <string>

// 3rd party
#include "sophus/se3.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"

// own
#include "vlslam.pb.h"

namespace feh {

/// \link: http://campar.in.tum.de/Main/StefanHinterstoisser
class LinemodDatasetLoader {
public:
    LinemodDatasetLoader(const std::string &dataroot);

    /// \brief: Grab datum at index i.
    /// \param i: index
    /// \param image: image at index i
    /// \param gm: ground truth model pose at index i
    /// \return: false if index out-of-range
    bool Grab(int i, cv::Mat &image, Sophus::SE3f &gm);

    int size() const { return size_; }
    std::vector<float> vertices() const { return vertices_; }
    std::vector<int> faces() const { return faces_; }

private:
    std::vector<float> vertices_;
    std::vector<int> faces_;
    std::vector<std::string> jpg_files_, trans_files_, rot_files_;
    std::string dataroot_;
    Sophus::SE3f transform_;
    int size_;

public:
    static constexpr float fx_ = 572.41140f;
    static constexpr float fy_ = 573.57043f;
    static constexpr float cx_ = 325.26110f;
    static constexpr float cy_ = 242.04899f;
    static constexpr float rows_ = 480;
    static constexpr float cols_ = 640;
};

class VlslamDatasetLoader {
public:
    VlslamDatasetLoader() {}
    VlslamDatasetLoader(const std::string &dataroot);
    /// \brief: Grab datum at index i.
    /// \param i: index
    /// \param image:
    /// \param edgemap:
    /// \param bboxlist:
    /// \param gwc: camera to world transformation
    /// \param Rg: gravity rotation
    virtual bool Grab(int i,
                      cv::Mat &image,
                      cv::Mat &edgemap,
                      vlslam_pb::BoundingBoxList &bboxlist,
                      Sophus::SE3f &gwc,
                      Sophus::SO3f &Rg);
    /// \param fullpath: full path to the image file
    virtual bool Grab(int i,
                      cv::Mat &image,
                      cv::Mat &edgemap,
                      vlslam_pb::BoundingBoxList &bboxlist,
                      Sophus::SE3f &gwc,
                      Sophus::SO3f &Rg,
                      std::string &fullpath);

    virtual int size() const { return size_; }
private:
    std::string dataroot_;
    vlslam_pb::Dataset dataset_;
    std::vector<std::string> png_files_, edge_files_, bbox_files_;
    int size_;
};

class ICLDatasetLoader : public VlslamDatasetLoader {
public:
    ICLDatasetLoader(const std::string &dataroot);

    bool Grab(int i,
              cv::Mat &image,
              cv::Mat &edgemap,
              vlslam_pb::BoundingBoxList &bboxlist,
              Sophus::SE3f &gwc,
              Sophus::SO3f &Rg) override ;
    /// \param fullpath: full path to the image file
    bool Grab(int i,
              cv::Mat &image,
              cv::Mat &edgemap,
              vlslam_pb::BoundingBoxList &bboxlist,
              Sophus::SE3f &gwc,
              Sophus::SO3f &Rg,
              std::string &fullpath) override;

    int size() const override { return size_; }
private:
    std::string dataroot_;
    vlslam_pb::Dataset dataset_;
    std::vector<std::string> png_files_, edge_files_, bbox_files_;
    std::vector<Sophus::SE3f> poses_;
    int size_;
};

/// \link: http://www.karlpauwels.com/datasets/rigid-pose/
class RigidPoseDatasetLoader {
public:
    enum class Tag : int {
        Left = 0x01,
        Right = 0x02,
        NoiseFree = 0x10,
        Noisy = 0x20,
        Occluded = 0x40
    };

    RigidPoseDatasetLoader(const std::string &dataroot,
                           const std::string &dataset,
                           int tag);
    ~RigidPoseDatasetLoader() {
        capture_.release();
    }
    bool Grab(cv::Mat &image, Sophus::SE3f &pose);
    Sophus::SE3f GetPose(int i) const;

    std::vector<float> vertices() const { return vertices_; }
    std::vector<int> faces() const { return faces_; }

private:
    std::string dataroot_, dataset_;
    cv::VideoCapture capture_;
    std::vector<float> vertices_;
    std::vector<int> faces_;
    std::vector<Eigen::Matrix<float, 6, 1>> g_;
    int index_;

public:
    static constexpr float focal_length_ = 500.6795; // in pixels
    static constexpr float baseline_ = 70.7722 * 1e-3; // in meters
    // principal points, a.k.a., nodal points
    static constexpr float cx_ = 352.1633; // column (in pixels)
    static constexpr float cy_ = 260.3113; // row (in pixels)
    static constexpr int rows_ = 480;
    static constexpr int cols_ = 640;
};

}   // namespace feh
