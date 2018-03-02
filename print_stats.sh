#!/bin/bash
ls  src/*.cpp include/*.h include/common/*.h src/common/*.cpp include/shaders/*.comp |  xargs wc -l
# ls src/*.cpp include/vins/*.h |  xargs wc -l
