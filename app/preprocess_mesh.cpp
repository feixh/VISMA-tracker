//
// Created by visionlab on 2/18/18.
//

#include "igl/readOBJ.h"
#include "igl/writePLY.h"
#include "igl/writeOBJ.h"
#include "glog/logging.h"


int main(int argc, char **argv) {
    CHECK(argc == 3) << "usage: preprocess_mesh PATH_TO_OBJ_FILE OUTPUT_PATH\n";
    Eigen::Matrix<float, Eigen::Dynamic, 3> TC, CN;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Vin;
    Eigen::Matrix<int, Eigen::Dynamic, 3> F, FTC, FN;
    igl::readOBJ(argv[1], Vin, TC, CN, F, FTC, FN);

    Eigen::Matrix<float, Eigen::Dynamic, 3> V = Vin.block(0, 0, Vin.rows(), 3);

    // center at origin
    V.rowwise() -= V.colwise().mean();

    // rotate around Y
    Eigen::AngleAxisf aa(-M_PI/2, Eigen::Vector3f::UnitY());
    V *= aa.toRotationMatrix().transpose();

    // normalize again
    V.rowwise() -= V.colwise().mean();

//    // flip Z and Y  for ShapeNet models
//    V.col(1) = -V.col(1);
//    V.col(2) = -V.col(2);

    Vin.block(0, 0, Vin.rows(), 3) = V;

    // write out mesh in canonical frame
    std::string basename = argv[2];
//    igl::writePLY(basename + ".ply", V, F); // only keey geometry in ply file
    igl::writeOBJ(basename + ".obj", Vin, F, CN, FN, TC, FTC);  // preserve color in obj file
}

