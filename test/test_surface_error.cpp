//
// Created by visionlab on 3/6/18.
//
// Test quantitative evaluation of surface reconstruction by the following procedure:
// 1) Densely sample a point cloud from the source mesh
// 2) For each point in the source point cloud, locate the closest triangle in the target mesh and
// 3) Compute the perpendicular distance from the point to the triangle
// 4) Compute statistics over all distances computed above.
// Use libigl AABB (Axis Aligned Bounding Box, an instantitation of quadtree) data structure for
// fast closest point query.

#include "eigen_alias.h"
// stl

// 3rd party
#include "igl/readOBJ.h"
#include "igl/AABB.h"
#include "Visualization/Visualization.h"
#include "Core/Core.h"

// feh
#include "geometry.h"

int main() {
    Eigen::Matrix<double, Eigen::Dynamic, 3> Vs, Vt;    // source & target vertices
    Eigen::Matrix<int, Eigen::Dynamic, 3> Fs, Ft;   // source & target faces
    std::string mesh_file = "/local/feixh/Dropbox/ECCV18/data/CAD_database/hermanmiller_aeron.obj";
    // LOAD AND PERTURBATE SOURCE MESH
    igl::readOBJ(mesh_file, Vs, Fs);
    for (int i = 0; i < Vs.rows(); ++i) {
        Vs(i, 0) += 0.1;
    }
    
    auto pts_s = feh::SamplePointCloudFromMesh(Vs, Fs, 50000);

    auto pc = std::make_shared<three::PointCloud>();
    pc->points_ = pts_s;
    three::DrawGeometries({pc}, "source point cloud");

    // LOAD AND CONSTRUCT DENSE POINT CLOUD
    igl::readOBJ(mesh_file, Vt, Ft);
//
    // CONSTRUCT AABB TREE
    igl::AABB<Eigen::Matrix<double, Eigen::Dynamic, 3>, 3> tree;
    tree.init(Vt, Ft);
    Eigen::Matrix<double, Eigen::Dynamic, 3> P = feh::StdVectorOfEigenVectorToEigenMatrix(pts_s); // query point list
    Eigen::VectorXd D2; // squared distance
    Eigen::VectorXi I;  // index into face list Ft
    Eigen::Matrix<double, Eigen::Dynamic, 3> C; // closest point in mesh, NOT necessarily a vertex
    // Given the face index I and coordinates of the closest point C, barycentric coordinates
    // of the closest point w.r.t the triangle can be recovered.
    tree.squared_distance(Vt, Ft, P, D2, I, C);
    auto D = D2.cwiseSqrt();
    std::cout << "mean distance=" << D.colwise().mean() << "\n";

    auto stats = feh::MeasureSurfaceError(Vs, Fs, Vt, Ft, folly::dynamic::object("num_samples", 50000));
    std::cout << "mean=" << stats.mean_ << "\n";
    std::cout << "std=" << stats.std_ << "\n";
    std::cout << "min=" << stats.min_ << "\n";
    std::cout << "max=" << stats.max_ << "\n";
    std::cout << "median=" << stats.median_ << "\n";
}

