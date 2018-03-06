//
// Created by visionlab on 3/2/18.
//
#include "tool.h"

// folly
#include "folly/FileUtil.h"
#include "folly/json.h"
#include "folly/Format.h"
// libigl
#include "igl/readOBJ.h"
#include <igl/writeOBJ.h>
#include "igl/readPLY.h"
// Open3D
#include "IO/IO.h"
#include "Visualization/Visualization.h"

// feh
#include "io_utils.h"
#include "geometry.h"

namespace feh {
void VisualizeResult(const folly::dynamic &config) {

    // EXTRACT PATHS
    std::string database_dir = config["CAD_database_root"].getString();

    std::string dataroot = config["dataroot"].getString();
    std::string dataset = config["dataset"].getString();
    std::string scene_dir = dataroot + "/" + dataset + "/";
    std::string fragment_dir = scene_dir + "/fragments/";
    // LOAD SCENE POINT CLOUD
    auto scene = std::make_shared<three::PointCloud>();

    // LOAD RESULT FILE
    std::string result_file = folly::sformat("{}/result.json", scene_dir);
    std::string contents;
    folly::readFile(result_file.c_str(), contents);
    folly::dynamic result = folly::parseJson(folly::json::stripComments(contents));
    // ITERATE AND GET THE LAST ONE
    auto packet = result.at(result.size() - 1);
    auto scene_est = std::make_shared<three::PointCloud>();
    std::unordered_map<int, Model> models_est;
    for (const auto &obj : packet) {
        auto pose = io::GetMatrixFromDynamic<double, 3, 4>(obj, "model_pose");
        std::cout << folly::format("id={}\nstatus={}\nshape={}\npose=\n",
                                   obj["id"].asInt(),
                                   obj["status"].asInt(),
                                   obj["model_name"].asString())
                  << pose << "\n";

        auto &this_model = models_est[obj["id"].asInt()];
        this_model.model_name_ = obj["model_name"].asString();
        this_model.model_to_scene_.block<3, 4>(0, 0) = pose;
        igl::readOBJ(folly::sformat("{}/{}.obj",
                                    database_dir,
                                    this_model.model_name_),
                     this_model.V_, this_model.F_);

        std::shared_ptr <three::PointCloud> model_pc = std::make_shared<three::PointCloud>();
        model_pc->points_ = SamplePointCloudFromMesh(
            this_model.V_, this_model.F_,
            config["visualization"]["model_samples"].asInt());
        model_pc->colors_.resize(model_pc->points_.size(), {255, 0, 0});
        model_pc->Transform(this_model.model_to_scene_);    // ALREADY IN CORVIS FRAME
        this_model.pcd_ptr_ = model_pc;

        *scene_est += *model_pc;
    }

    three::DrawGeometries({scene_est}, "reconstructed scene");
}

void VisualizeGroundTruth(const folly::dynamic &config) {
    std::string database_dir = config["CAD_database_root"].getString();

    std::string dataroot = config["dataroot"].getString();
    std::string dataset = config["dataset"].getString();
    std::string scene_dir = dataroot + "/" + dataset + "/";
    std::string fragment_dir = scene_dir + "/fragments/";
    // LOAD SCENE POINT CLOUD
    auto scene = std::make_shared<three::PointCloud>();
    three::ReadPointCloudFromPLY(scene_dir + "/test.klg.ply", *scene);

    std::vector<Eigen::Matrix<double, 6, 1>> vertices;
    std::vector<Eigen::Matrix<int, 3, 1>> faces;


    // LOAD RESULT FILE
    std::string alignment_file = fragment_dir + "/alignment.json";
    std::string contents;
    folly::readFile(alignment_file.c_str(), contents);
    folly::dynamic alignment = folly::parseJson(folly::json::stripComments(contents));

    int vertex_counter(0);
    for (const auto &obj : alignment.keys()) {
        std::string obj_name = obj.asString();
        std::string model_name = obj_name.substr(0, obj_name.find_last_of('_'));
        auto pose = io::GetMatrixFromDynamic<double, 3, 4>(alignment, obj_name);
        std::cout << obj_name << "\n" << model_name << "\n" << pose << "\n";
        // LOAD MESH
        Eigen::Matrix<double, Eigen::Dynamic, 6> v;
        Eigen::Matrix<int, Eigen::Dynamic, 3> f;
        igl::readOBJ(folly::sformat("{}/{}.obj", database_dir, model_name), v, f);
        std::cout << "v.size=" << v.rows() << "x" << v.cols() << "\n";
        // TRANSFORM TO SCENE FRAME
        v.leftCols(3) = (v.leftCols(3) * pose.block<3,3>(0,0).transpose()).rowwise() + pose.block<3, 1>(0, 3).transpose();
        for (int i = 0; i < v.rows(); ++i) {
            vertices.push_back(v.row(i));
        }
        for (int i = 0; i < f.rows(); ++i) {
            faces.push_back({vertex_counter + f(i, 0), vertex_counter + f(i, 1), vertex_counter + f(i, 2)});
        }
        // UPDATE VERTEX INDEX OFFSET
        vertex_counter += v.rows();
        // CONSTRUCT SCENE MESH BY APPENDING OBJECT MESH TO SCENE MESH
    }
    igl::writeOBJ("test.obj",
                  StdVectorOfEigenVectorToEigenMatrix(vertices),
                  StdVectorOfEigenVectorToEigenMatrix(faces));
}

}
