#!/bin/bash
ROOT=$PWD

cd thirdparty/zmqpp
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=..
make install

cd $ROOT
mkdir build
cd build
cmake ..
make -j
