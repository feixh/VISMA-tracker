//
// Created by visionlab on 2/19/18.
//
#pragma once

#include "common/eigen_alias.h"
#include <vector>
#include <random>
#include <chrono>

// 3rd party
#include "igl/AABB.h"
#include "folly/dynamic.h"

namespace feh {

/// \brief: Find the normal vector of a (hyper)plane given as a set of points.
template <typename T, int DIM=3>
Eigen::Matrix<T, DIM, 1> FindPlaneNormal(const Eigen::Matrix<T, Eigen::Dynamic, DIM> &pts) {
    auto pts_n = pts.rowwise() - pts.colwise().mean();
    auto P = pts_n.transpose() * pts_n / pts.rows();
    Eigen::JacobiSVD<Eigen::Matrix<T, DIM, DIM>> svd(P, Eigen::ComputeThinU | Eigen::ComputeFullV);
    Eigen::Matrix<T, DIM, 1> n = svd.matrixV().col(2);
    n.normalize();
    return n;
}

/// \brief: Convert from std vector of Eigen vectors to Eigen matrix.
template <typename T, int DIM=3>
Eigen::Matrix<T, Eigen::Dynamic, DIM>
StdVectorOfEigenVectorToEigenMatrix(const std::vector<Eigen::Matrix<T, DIM, 1> > &v) {
    Eigen::Matrix<T, Eigen::Dynamic, DIM> out;
    out.resize(v.size(), DIM);
    for (int i = 0; i < v.size(); ++i) {
        out.row(i) = v[i];
    }
    return out;
}

/// \brief: Convert from an Eigen matrix type to a std vector of Eigen vectors.
template <typename T, int DIM=3>
std::vector<Eigen::Matrix<T, DIM, 1>>
EigenMatrixToStdVectorOfEigenVector(const Eigen::Matrix<T, Eigen::Dynamic, DIM> &m) {
    std::vector<Eigen::Matrix<T, DIM, 1>> out(m.rows());
    for (int i = 0; i < m.rows(); ++i) out[i] = m.row(i);
    return out;
};

/// \brief: Convert from a std vector of Eigen vectors to an Eigen matrix.
template <typename T, int DIM=3>
Eigen::Matrix<T, DIM, 1> StdVectorOfEigenVectorMean(const std::vector<Eigen::Matrix<T, DIM, 1>> &v) {
    return StdVectorOfEigenVectorToEigenMatrix(v).colwise().mean();
};

/// \brief: Compute the rotation matrix to align two vectors.
template <typename T>
Eigen::Matrix<T, 3, 3> RotationBetweenVectors(Eigen::Matrix<T, 3, 1> u, Eigen::Matrix<T, 3, 1> v) {
    return Eigen::Quaternion<T>::FromTwoVectors(u, v).toRotationMatrix();
};

/// \brief: Sample point cloud from surface uniformly.
template <typename T>
std::vector<Eigen::Matrix<T, 3, 1>>
SamplePointCloudFromMesh(const Eigen::Matrix<T, Eigen::Dynamic, 3> &V,
                         const Eigen::Matrix<int, Eigen::Dynamic, 3> &F,
                         int max_num_pts = 1000) {
    // FIRST COMPUTE TOTAL SURFACE AREA
    std::vector<T> area(F.rows());
    T total_area(0);
    for (int i = 0; i < F.rows(); ++i ) {
        std::array<Eigen::Matrix<T, 3, 1>, 3> v{V.row(F(i, 0)), V.row(F(i, 1)), V.row(F(i, 2))};
        area[i] = 0.5*(v[1]-v[0]).cross(v[2]-v[0]).norm();
        total_area += area[i];
    }
    area[0] /= total_area;
    for (int i = 1; i < area.size(); ++i) {
        area[i] = area[i-1] + area[i] / total_area;
    }

    std::knuth_b generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<T> uniform(0, 1);
    std::vector<Eigen::Matrix<T, 3, 1>> out;
    for (int i = 0; i < max_num_pts; ++i) {
        T r = uniform(generator);
        for (int k = 0; k < area.size()-1; ++k) {
            if (area[k] <= r && r < area[k+1]) {
                std::array<Eigen::Matrix<T, 3, 1>, 3> v{V.row(F(k, 0)), V.row(F(k, 1)), V.row(F(k, 2))};
                T a = uniform(generator);
                T b = uniform(generator);  // barycentric coordinates
                out.push_back(v[0] + a * (v[1]-v[0]) + b * (v[2]-v[0]));
            }

        }

    }
    return out;
};



////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////
// OCTREE NODE
////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////
template <typename T, int L>
struct OctreeNode {
    OctreeNode(const Eigen::Matrix<T, 3, 1> &min, const Eigen::Matrix<T, 3, 1> &max, int level):
        min_(min), max_(max), mid_(0.5*(min_+max_)), level_(level) {
        for (uint8_t i = 0; i < 8; ++i) node_[i] = nullptr;
    }
    ~OctreeNode() {
        for (uint8_t i = 0; i < 8; ++i) delete node_[i];
    }

    void Insert(int index, const Eigen::Matrix<T, 3, 1> &pt) {
//        index_.push_back(index);
        if (level_ < L) {
            uint8_t code = (pt[0] < mid_[0]) | ((pt[1] < mid_[1]) << 1) | ((pt[2] < mid_[2]) << 2);
            if (node_[code] == nullptr) {
                // CREATE NEW NODE
                node_[code] = new OctreeNode<T, L>(
                    {code & 1 ? min_[0] : mid_[0], code & 2 ? min_[1] : mid_[1], code & 3 ? min_[2] : mid_[2]},
                    {code & 1 ? mid_[0] : max_[0], code & 1 ? mid_[1] : max_[1], code & 1 ? mid_[2] : max_[2]},
                    level_+1);
            }
            node_[code]->Insert(index, pt);
        } else {
            // LEAF NODE, STORE INDEX
            index_.push_back(index);
        }
    }

    std::vector<int> Find(const Eigen::Matrix<T, 3, 1> &pt) {
        if (level_ == L) return index_;
        uint8_t code = (pt[0] < mid_[0]) | ((pt[1] < mid_[1]) << 1) | ((pt[2] < mid_[2]) << 2);
        return node_[code]->Find(pt);
    }

    Eigen::Matrix<T, 3, 1> min_, max_, mid_;
    int level_;
    std::array<OctreeNode<T, L>*, 8> node_;
    std::vector<int> index_;
};

////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////
// OCTREE
////////////////////////////////////////
////////////////////////////////////////
////////////////////////////////////////
template<typename T, int L>
class Octree {
public:
    using Type = T;

    Octree(const Eigen::Matrix<T, Eigen::Dynamic, 3> &pts);
    Octree(const std::vector<Eigen::Matrix<T, 3, 1>> &pts);
    ~Octree();
    const Eigen::Matrix<T, 3, 1> &Resolution() const { return res_; };
    int Find(const Eigen::Matrix<T, 3, 1> &pt) {
        if (root_ == nullptr) return -1;
        auto set = root_->Find(pt);
        T min_dist = std::numeric_limits<T>::max();
        int best_index = -1;
        for (int index : set) {
            T dist = (pts_[index]-pt).norm();
            if (dist < min_dist) {
                best_index = index;
                min_dist = dist;
            }
        }
        return best_index;
    }

private:
    Eigen::Matrix<T, 3, 1> min_, max_;    // bounds and mid-point
    Eigen::Matrix<T, 3, 1> res_;
    std::vector<Eigen::Matrix<T, 3, 1>> pts_;
    OctreeNode<T, L> *root_;
};

template<typename T,int L>
Octree<T, L>::Octree(const Eigen::Matrix<T, Eigen::Dynamic, 3> &pts):
    Octree(EigenMatrixToStdVectorOfEigenVector(pts)) {

}

template<typename T, int L>
Octree<T, L>::Octree(const std::vector<Eigen::Matrix<T, 3, 1>> &pts):
    pts_(pts),
    root_(nullptr)
{
    // INITIALIZE BOUND
    min_ << std::numeric_limits<T>::max(), std::numeric_limits<T>::max(), std::numeric_limits<T>::max();
    max_ << std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest();
    for (const auto &pt : pts) {
        for (int i = 0; i < 3; ++i) {
            if (pt[i] < min_[i]) min_[i] = pt[i];
            if (pt[i] > max_[i]) max_[i] = pt[i];
        }
    }
    res_ = (max_ - min_) / (1 << L);
    root_ = new OctreeNode<T, L>(min_, max_, 0);
    for (int i = 0; i < pts.size(); ++i) {
        const auto &pt = pts[i];
        root_->Insert(i, pt);
    }
}

template<typename T, int L>
Octree<T, L>::~Octree() {
    delete root_;
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// QUANTITATIVE SURFACE ERROR METRIC
// reference: A benchmark for RGB-D visual odometry, 3D reconstruction and SLAM
// Input: Two meshes
// 1) Densely sample a point cloud from the source mesh
// 2) For each point in the source point cloud, locate the closest triangle in the target mesh and
// 3) Compute the perpendicular distance from the point to the triangle
// 4) Compute statistics over all distances computed above.
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename T>
struct SurfaceErrorMetric {
    T mean_, std_, median_, min_, max_;
};
/// \brief: Measure surface error.
/// \param Vs: Source vertices.
/// \param Fs: Source faces.
/// \param Vt: Target vertices.
/// \param Ft: Target faces.
template <typename T>
SurfaceErrorMetric<T> MeasureSurfaceError(
    const Eigen::Matrix<T, Eigen::Dynamic, 3> &Vs,
    const Eigen::Matrix<int, Eigen::Dynamic, 3> &Fs,
    const Eigen::Matrix<T, Eigen::Dynamic, 3> &Vt,
    const Eigen::Matrix<int, Eigen::Dynamic, 3> &Ft,
    const folly::dynamic &options) {
    // DENSELY SAMPLE FROM THE INPUT MESH
    auto pts_s = SamplePointCloudFromMesh(Vs, Fs, options["num_samples"].getInt());
    // CONSTRUCT AABB TREE FROM TARGET MESH
    igl::AABB<Eigen::Matrix<T, Eigen::Dynamic, 3>, 3> tree;
    tree.init(Vt, Ft);
    Eigen::Matrix<T, Eigen::Dynamic, 3> P = feh::StdVectorOfEigenVectorToEigenMatrix(pts_s); // query point list
    Eigen::VectorXd D2; // squared distance
    Eigen::VectorXi I;  // index into face list Ft
    Eigen::Matrix<T, Eigen::Dynamic, 3> C; // closest point in mesh, NOT necessarily a vertex
    // Given the face index I and coordinates of the closest point C, barycentric coordinates
    // of the closest point w.r.t the triangle can be recovered.
    tree.squared_distance(Vt, Ft, P, D2, I, C);
    auto D = D2.cwiseSqrt();

    SurfaceErrorMetric<T> out{0, 0, 0,
                              std::numeric_limits<T>::max(),
                              std::numeric_limits<T>::lowest()};
    std::vector<T> tmp(D.size());
    for (int i = 0; i < D.size(); ++i) {
        out.mean_ += D(i);
        out.std_ += D(i) * D(i);
        out.min_ = std::min(out.min_, D(i));
        out.max_ = std::max(out.max_, D(i));
        tmp[i] = D(i);
    }
    out.mean_ /= D.size();
    out.std_ = sqrt(out.std_ / D.size() - out.mean_ * out.mean_);
    std::sort(tmp.begin(), tmp.end());
    out.median_ = tmp[tmp.size() >> 1];
    return out;
}


}   // namespace feh
