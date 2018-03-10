//
// Created by visionlab on 3/9/18.
//
#include "geometry.h"
#include "igl/readOBJ.h"

int main(int argc, char **argv) {
    Eigen::Matrix<double, Eigen::Dynamic, 3> V;
    Eigen::Matrix<int, Eigen::Dynamic, 3> F;
    igl::readOBJ(argv[1], V, F);
    auto n = feh::FindPlaneNormal(V);
    std::cout << n.transpose() << "\n";
}

