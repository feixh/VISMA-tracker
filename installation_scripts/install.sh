#!/bin/bash
project_root=$(pwd)
echo $project_root

cd $project_root/thirdparty/jsoncpp
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=.. -DBUILD_SHARED_LIBS=ON
make -j
make install

cd $project_root/thirdparty/zmqpp
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=..
make -j
make install

cd $project_root/thirdparty/lcm-1.4.0
mkdir build
cd build
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
