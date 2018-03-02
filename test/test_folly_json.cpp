//
// Created by visionlab on 2/17/18.
//
// Play with json parser and dynamic from facebook folly library.

#include "folly/json.h"
#include "folly/FileUtil.h"
#include <iostream>

int main() {
    std::string json_content;
    folly::readFile("../cfg/chair_tracker.json", json_content);
    folly::dynamic config = folly::parseJson(folly::json::stripComments(json_content));
    std::cout << config["CAD_database_root"].asString() << "\n";
//    std::cout << config["not_existing_key"].asInt() << "\n";
    std::cout << config["visualization"]["show_bounding_boxes"].asBool();
}

