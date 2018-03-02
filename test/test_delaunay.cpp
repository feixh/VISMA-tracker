#include <iostream>

#include "delaunay.h"
#include <Eigen/Dense>

int main(int argc, char *argv[])
{

// Input polygon
    Eigen::MatrixXf V;

// Triangulated interior
    Eigen::MatrixXf V2;
    Eigen::MatrixXi F2;

    // Create the boundary of a square
    V.resize(15,2);
//    E.resize(8,2);
//    H.resize(1,2);

    V <<  0,       0,
        -0.416 ,  0.909   ,
        -1.35  ,  0.436   ,
        -1.64  , -0.549   ,
        -1.31  , -1.51    ,
        -0.532 , -2.17    ,
        0.454 , -2.41    ,
        1.45  , -2.21    ,
        2.29  , -1.66    ,
        2.88 ,  -0.838  ,
        3.16 ,   0.131  ,
        3.12 ,   1.14   ,
        2.77 ,   2.08   ,
        2.16 ,   2.89   ,
        1.36 ,   3.49;

//    E << 0,1, 1,2, 2,3, 3,0,
//        4,5, 5,6, 6,7, 7,4;
//
//    H << 0,0;

    // Triangulate the interior
    feh::Delaunay::Triangulate(V, V2, F2);

    std::cout << "vertices=\n";
    for (int i = 0; i < V2.rows(); ++i) {
        std::cout << V2.row(i) << "\n";
    }

    std::cout << "faces=\n";
    for (int i = 0; i < F2.rows(); ++i) {
        std::cout << F2.row(i) << "\n";
    }

//    // Plot the generated mesh
//    igl::viewer::Viewer viewer;
//    viewer.data.set_mesh(V2,F2);
//    viewer.launch();
    cv::Mat display = feh::Delaunay::BuildView(V2, F2);
    cv::imshow("delaunay", display);
    cv::waitKey();
}
