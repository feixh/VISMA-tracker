//
// Created by visionlab on 2/19/18.
//
// test my octree implementation
#include "geometry.h"
#include <iostream>

int LinearSearch(const Eigen::Matrix<double, Eigen::Dynamic, 3> &pts, const Eigen::Matrix<double, 3, 1> &q) {
    int index = 0;
    double min_dist = (pts.row(index) - q.transpose()).norm();
    for (int i = 1; i < pts.rows(); ++i) {
        double dist = (pts.row(i) - q.transpose()).norm();
        if ( dist < min_dist) {
            min_dist = dist;
            index = i;
        }
    }
    return index;
}

int main() {
    Eigen::Matrix<double, Eigen::Dynamic, 3> pts;
    pts.resize(50000, 3);
    pts.setRandom();
    pts *= 10;
    feh::Octree<double, 10> tree(pts);
    std::cout << "res=" << tree.Resolution().transpose() << "\n";
    {
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < pts.rows(); ++i) {
            Eigen::Matrix<double, 3, 1> pt = pts.row(i);
            int index1 = LinearSearch(pts, pt);
            assert(i == index1);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "linear search: " << std::chrono::duration<float>(t2-t1).count() << "\n";

    }

    {
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < pts.rows(); ++i) {
            Eigen::Matrix<double, 3, 1> pt = pts.row(i);
            int index2 = tree.Find(pt);
            assert(i == index2);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::cout << "octree search: " << std::chrono::duration<float>(t2-t1).count() << "\n";

    }
}

