//
// Created by feixh on 10/17/17.

#include "io_utils.h"

int main() {
    std::string inputfile = "../resources/swivel_chair.obj";
    std::vector<float> vertices;
    std::vector<int> faces;
    feh::io::LoadMeshFromObjFile(inputfile, vertices, faces);
}
