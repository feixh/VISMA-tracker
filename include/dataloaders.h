//
// Created by feixh on 11/15/17.
//
#pragma once

// stl
#include <vector>
#include <string>
#include <unordered_map>

// 3rd party
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"
#include "glog/logging.h"

// own
#include "vlslam.pb.h"
#include "alias.h"
#include "se3.h"

namespace feh {

/// \brief: Load edgemap from protobuf file.
bool LoadEdgeMap(const std::string &filename, cv::Mat &edge);
/// \brief: Load a list of mesh file paths.
/// \param root: Root directory of the CAD database. All the meshes are put directly under this directory.
/// \param cat_json: Json file of the list of meshes of a certain category.
std::vector<std::string> LoadMeshDatabase(const std::string &root, const std::string &cat_json);

/// \brief: Convert protobuf repeated field to Eigen matrix.
Mat4f SE3FromArray(float *data);
Mat4f SE3FromArray(double *data);

/// \link: http://campar.in.tum.de/Main/StefanHinterstoisser
class LinemodDatasetLoader {
public:
    LinemodDatasetLoader(const std::string &dataroot);

    /// \brief: Grab datum at index i.
    /// \param i: index
    /// \param image: image at index i
    /// \param gm: ground truth model pose at index i
    /// \return: false if index out-of-range
    bool Grab(int i, cv::Mat &image, SE3 &gm);

    int size() const { return size_; }
    MatXf vertices() const { return vertices_; }
    MatXi faces() const { return faces_; }

private:
    MatXf vertices_;
    MatXi faces_;
    std::vector<std::string> jpg_files_, trans_files_, rot_files_;
    std::string dataroot_;
    SE3 transform_;
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
                      SE3 &gwc,
                      SO3 &Rg);
    /// \param fullpath: full path to the image file
    virtual bool Grab(int i,
                      cv::Mat &image,
                      cv::Mat &edgemap,
                      vlslam_pb::BoundingBoxList &bboxlist,
                      SE3 &gwc,
                      SO3 &Rg,
                      std::string &fullpath);
    std::unordered_map<int64_t, std::array<double, 6>> GrabPointCloud(int i, const cv::Mat &img);

    virtual int size() const { return size_; }
protected:
    std::string dataroot_;
    vlslam_pb::Dataset dataset_;
    std::vector<std::string> png_files_, edge_files_, bbox_files_;
    int size_;
};

class KittiDatasetLoader : public VlslamDatasetLoader{
public:
    KittiDatasetLoader(const std::string &dataroot);
    bool Grab(int i,
              cv::Mat &image,
              cv::Mat &edgemap,
              vlslam_pb::BoundingBoxList &bboxlist,
              SE3 &gwc,
              SO3 &Rg) override;

    bool Grab(int i,
              cv::Mat &image,
              cv::Mat &edgemap,
              vlslam_pb::BoundingBoxList &bboxlist,
              SE3 &gwc,
              SO3 &Rg,
              std::string &fullpath) override;
private:
    std::vector<SE3> poses_;
};

class ICLDatasetLoader : public VlslamDatasetLoader {
public:
    ICLDatasetLoader(const std::string &dataroot);

    bool Grab(int i,
              cv::Mat &image,
              cv::Mat &edgemap,
              vlslam_pb::BoundingBoxList &bboxlist,
              SE3 &gwc,
              SO3 &Rg) override ;
    /// \param fullpath: full path to the image file
    bool Grab(int i,
              cv::Mat &image,
              cv::Mat &edgemap,
              vlslam_pb::BoundingBoxList &bboxlist,
              SE3 &gwc,
              SO3 &Rg,
              std::string &fullpath) override;
private:
    std::vector<SE3> poses_;
};

class SceneNNDatasetLoader : public VlslamDatasetLoader {
public:
    SceneNNDatasetLoader(const std::string &dataroot);
    bool Grab(int i,
              cv::Mat &image,
              cv::Mat &edgemap,
              vlslam_pb::BoundingBoxList &bboxlist,
              SE3 &gwc,
              SO3 &Rg) override ;
    /// \param fullpath: full path to the image file
    bool Grab(int i,
              cv::Mat &image,
              cv::Mat &edgemap,
              vlslam_pb::BoundingBoxList &bboxlist,
              SE3 &gwc,
              SO3 &Rg,
              std::string &fullpath) override;
private:
    std::vector<SE3> poses_;
    int skip_head_, until_last_;
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
    bool Grab(cv::Mat &image, SE3 &pose);
    SE3 GetPose(int i) const;

    MatXf vertices() const { return vertices_; }
    MatXi faces() const { return faces_; }

private:
    std::string dataroot_, dataset_;
    cv::VideoCapture capture_;
    MatXf vertices_;
    MatXi faces_;
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
