#!/bin/bash
project_root=$(pwd)
echo $project_root

mkdir -p build
cd build
cmake ..
make -j
cd $project_root
#sudo make install
#python setup.py install --user
#python test_camera.py
