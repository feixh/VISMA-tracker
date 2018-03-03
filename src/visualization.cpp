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
// Open3D
#include "IO/IO.h"
#include "Visualization/Visualization.h"

// feh
#include "io_utils.h"
#include "geometry.h"

namespace feh {
void VisualizationTool(const folly::dynamic &config) {

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

//    auto ret = RegisterScenes(models, models_est);
//    auto T_ef_corvis = ret.transformation_;
//    std::cout << "T_ef_corvis=\n" << T_ef_corvis << "\n";
//    for (int i = 0; i < ret.correspondence_set_.size(); ++i) {
//        std::cout << folly::format("{}-{}\n", ret.correspondence_set_[i][0], ret.correspondence_set_[i][1]);
//    }
//
//    if (config["evaluation"]["ICP_refinement"].asBool()) {
//        // RE-LOAD THE SCENE
//        std::shared_ptr<three::PointCloud> raw_scene = std::make_shared<three::PointCloud>();
//        three::ReadPointCloudFromPLY(scene_dir + "/test.klg.ply", *raw_scene);
//        // FIXME: MIGHT NEED CROP THE 3D REGION-OF-INTEREST HERE
//        auto result = ICPRefinement(raw_scene,
//                                    models_est,
//                                    T_ef_corvis,
//                                    config["evaluation"]);
//        T_ef_corvis = result.transformation_;
//    }
//
////    three::ReadPointCloudFromPLY(config["scene_directory"].getString() + "/test.klg.ply", *scene);
//    // NOW LETS LOOK AT THE ESTIMATED SCENE IN RGB-D SCENE FRAME
//    for (const auto &kv : models_est) {
//        const auto &this_model = kv.second;
//        this_model.pcd_ptr_->Transform(T_ef_corvis);
//        *scene += *(this_model.pcd_ptr_);
//    }
//    three::DrawGeometries({scene});
//    three::WritePointCloud(fragment_dir+"/augmented_view.ply", *scene);
}


}
