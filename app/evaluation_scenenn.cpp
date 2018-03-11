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

int main(int argc, char **argv) {
    std::string contents;
    folly::readFile("../cfg/scenenn.json", contents);
    folly::dynamic config = folly::parseJson(folly::json::stripComments(contents));
    std::string dataroot = config["dataroot"].getString();
    std::string dataset  = config["dataset"].getString();
    if (argc > 1) dataset = argv[1];    // overwrite dataset
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
    std::vector<std::pair<MatX3d, MatX3i>> vf;
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
        vf.push_back(std::make_pair(v, f));
    }

    //
    std::vector<int> support_set;
    int max_support_size;
    double best_floor;
    for (int i = 0; i < vf.size(); ++i) {
        double floor = vf[i].first.col(1).minCoeff();
        int support_size = 1;
        for (int j = 0; j < vf.size(); ++j) {
            if (i != j && fabs(vf[j].first.col(1).minCoeff()-floor) < 0.2) {
                ++support_size;
            }
        }
        if (support_size > max_support_size) {
            max_support_size = support_size;
            best_floor = floor;
        }
    }

    for (int k = 0; k < vf.size(); ++k) {
        auto v = vf[k].first;
        auto f = vf[k].second;
        int v_offset = vertices.size();
        if (fabs(v.col(1).minCoeff()-best_floor) < 0.2) {
            for (int i = 0; i < v.rows(); ++i) {
                vertices.push_back(v.row(i));
            }
            for (int i = 0; i < f.rows(); ++i) {
                faces.push_back({v_offset + f(i, 0), v_offset + f(i, 1), v_offset + f(i, 2)});
            }
        }
    }

    auto Vr = StdVectorOfEigenVectorToEigenMatrix(vertices);
    auto Fr = StdVectorOfEigenVectorToEigenMatrix(faces);

    // dump assembled mesh
    std::cout << TermColor::cyan << "writing result.obj" << TermColor::endl;
    igl::writeOBJ(dataroot + "/result.obj", Vr, Fr);

    // compare reconstructed mesh to ground truth mesh
    folly::dynamic options = folly::dynamic::object("num_samples", 50000);
    std::cout << TermColor::cyan << "computing surface error " << TermColor::endl;
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

