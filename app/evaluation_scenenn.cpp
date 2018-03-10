//
// Created by visionlab on 3/10/18.
//
#include "common/eigen_alias.h"
#include "geometry.h"
#include "io_utils.h"

// stl
#include <sstream>

// 3rd party
#include "folly/dynamic.h"
#include "folly/json.h"
#include "folly/FileUtil.h"
#include "igl/readOBJ.h"
#include "igl/readPLY.h"
#include "igl/writeOBJ.h"

using namespace feh;

std::list<std::array<double, 10>> GetOBBoxFromAnnotation(const folly::dynamic &annotation) {
    std::list<std::array<double, 10>> obbox_list;
    folly::dynamic labels = annotation["annotation"]["label"];
    for (auto each : labels) {
        bool is_chair = false;
        if (each.getDefault("-text", "invalid") == "chair") {
            std::cout << each["-obbox"].getString() << "\n";
            std::array<double, 10> obbox;
            std::istringstream(each["-obbox"].getString()) >> obbox[0] >> obbox[1] >> obbox[2]
                                                           >> obbox[3] >> obbox[4] >> obbox[5]
                                                           >> obbox[6] >> obbox[7] >> obbox[8]
                                                           >> obbox[9];
            obbox_list.push_back(obbox);
        }
    }
    return obbox_list;
}

int main() {
    std::string contents;
    folly::readFile("../cfg/scenenn.json", contents);
    folly::dynamic config = folly::parseJson(folly::json::stripComments(contents));
    std::string dataroot = config["dataroot"].getString();
    std::string dataset  = config["dataset"].getString();
    dataroot = dataroot + "/" + dataset + "/";
    std::string database_dir = dataroot;

//    // load annotation
//    folly::readFile((dataroot+dataset+".json").c_str(), contents);
//    GetOBBoxFromAnnotation(folly::parseJson((folly::json::stripComments(contents))));
//    exit(0);

    // load ground truth scene mesh
    MatX3d Vg;
    MatX3i Fg;
    igl::readOBJ(dataroot+dataset+"_objects.obj", Vg, Fg);

    // save out ground truth for inspection
    igl::writeOBJ(dataroot+"tmp.obj", Vg, Fg);

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

