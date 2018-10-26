#!/bin/bash
project_root=$(pwd)
echo $project_root

mkdir -p thirdparty/jsoncpp/build
cd thirdparty/jsoncpp/build
cmake .. -DCMAKE_INSTALL_PREFIX=.. -DBUILD_SHARED_LIBS=TRUE
make -j
make install
cd $project_root


mkdir -p thirdparty/fmt/build
cd thirdparty/fmt/build
cmake .. -DCMAKE_INSTALL_PREFIX=.. -DBUILD_SHARED_LIBS=TRUE
make -j
make install
cd $project_root

mkdir -p thirdparty/Sophus/build
cd thirdparty/Sophus/build
cmake .. -DCMAKE_INSTALL_PREFIX=..
make -j
make install
cd $project_root


mkdir -p build
cd build
cmake ..
make -j
cd $project_root
#sudo make install
#python setup.py install --user
#python test_camera.py
