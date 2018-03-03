//
// Created by visionlab on 2/19/18.
//
// Tools for ground truth annotation and evaluation.

#include "common/eigen_alias.h"

// stl
#include <chrono>
#include <sophus/se3.hpp>

// 3RD PARTY
// folly
#include "folly/dynamic.h"
#include "folly/json.h"
#include "folly/FileUtil.h"
// igl
#include "igl/readOBJ.h"

// feh
#include "io_utils.h"
#include "tool.h"

using namespace feh;

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        std::cout << "USAGE:\n tool OPTION [DATASET]\n OPTION=a|e\n a for annotation\n e for evaluation\n DATASET=dataset to evaluate";
        exit(-1);
    }
    // READ IN CONFIGURATION
    folly::fbstring contents;
    folly::readFile("../cfg/tool.json", contents);
    folly::dynamic config = folly::parseJson(folly::json::stripComments(contents));
    if (argc == 3) {
        config["dataset"] = std::string(argv[2]);
    }
    if (argv[1][0] == 'a') {
        AnnotationTool(config);
    } else if (argv[1][0] == 'e') {
        EvaluationTool(config);
    } else if (argv[1][0] == 'v') {
        VisualizationTool(config);
    }
    else {
        std::cout << "USAGE:\n tool OPTION\n OPTION=[a|e|v]\n a for annotation\n e for evaluation\n v for visualization";
        exit(-1);
    }

}

