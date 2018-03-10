//
// Created by visionlab on 3/10/18.
//
#include "common/eigen_alias.h"
#include "geometry.h"
#include "io_utils.h"

// 3rd party
#include "folly/dynamic.h"
#include "folly/json.h"
#include "folly/FileUtil.h"
#include "igl/readOBJ.h"
#include "igl/readPLY.h"
#include "igl/writeOBJ.h"

using namespace feh;

int main() {
    std::string contents;
    folly::readFile("../cfg/scenenn.json", contents);
    folly::dynamic config = folly::parseJson(folly::json::stripComments(contents));
    std::string dataroot = config["dataroot"].getString();
    std::string dataset  = config["dataset"].getString();
    dataroot = dataroot + "/" + dataset + "/";
    std::string database_dir = dataroot;

    // load ground truth scene mesh
    MatX3d Vg;
    MatX3i Fg;
    igl::readPLY(dataroot+dataset+".ply", Vg, Fg);

    // locate the object poses
    folly::readFile((dataroot+"result.json").c_str(), contents);
    folly::dynamic result = folly::parseJson(folly::json::stripComments(contents));
    auto packet = result.at(result.size()-1);
    std::list<std::pair<std::string, Eigen::Matrix<double, 3, 4>>> objects;
    for (const auto &obj : packet) {
        auto pose = io::GetMatrixFromDynamic<double, 3, 4>(obj, "model_pose");
        std::cout << folly::format("id={}\nstatus={}\nshape={}\npose=\n",
                                   obj["id"].asInt(),
                                   obj["status"].asInt(),
                                   obj["model_name"].asString())
                  << pose << "\n";
        std::string model_name = obj["model_name"].asString();
        objects.push_back(std::make_pair(model_name, pose));
    }


    // assemble result scene mesh
    std::vector<Vec3d> vertices;
    std::vector<Vec3i> faces;
    for (const auto &obj : objects) {
        std::string model_name = obj.first;
        auto pose = obj.second;
        // LOAD MESH
        Eigen::Matrix<double, Eigen::Dynamic, 3> v;
        Eigen::Matrix<int, Eigen::Dynamic, 3> f;
        igl::readOBJ(folly::sformat("{}/{}.obj", database_dir, model_name), v, f);
        std::cout << "v.size=" << v.rows() << "x" << v.cols() << "\n";
        // TRANSFORM TO SCENE FRAME
        v.leftCols(3) = (v.leftCols(3) * pose.block<3,3>(0,0).transpose()).rowwise() + pose.block<3, 1>(0, 3).transpose();
//        v.leftCols(3) = (v.leftCols(3) * alignment.block<3,3>(0,0).transpose()).rowwise() + alignment.block<3, 1>(0, 3).transpose();
        int v_offset = vertices.size();
        for (int i = 0; i < v.rows(); ++i) {
            vertices.push_back(v.row(i));
        }
        for (int i = 0; i < f.rows(); ++i) {
            faces.push_back({v_offset + f(i, 0), v_offset + f(i, 1), v_offset + f(i, 2)});
        }
    }

    auto Vr = StdVectorOfEigenVectorToEigenMatrix(vertices);
    auto Fr = StdVectorOfEigenVectorToEigenMatrix(faces);

    // dump assembled mesh
    igl::writeOBJ(dataroot + "/result.obj", Vr, Fr);

    // compare reconstructed mesh to ground truth mesh
    folly::dynamic options = folly::dynamic::object("num_samples", 500000);
    auto stats = MeasureSurfaceError(Vr, Fr, Vg, Fg, options);

    // save quantitative results
    folly::dynamic quat = folly::dynamic::object
        ("mean", stats.mean_)
        ("std", stats.std_)
        ("min", stats.min_)
        ("max", stats.max_)
        ("median", stats.median_);
    std::string quant_file = dataroot + "surface_error.json";
    folly::writeFile(folly::toPrettyJson(quat), quant_file.c_str());

    // print results here
    std::cout << folly::toPrettyJson(quat) << "\n";
}

