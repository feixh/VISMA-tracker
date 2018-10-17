cmake_minimum_required(VERSION 3.8)

#project(simplerender LANGUAGES CXX CUDA)
project(vlslam_plus)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMakeModules)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall -g -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare -Wno-unused-function -Wno-unused-variable -Wno-narrowing")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )
#set(CMAKE_CUDACXX_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES  /usr/local/cuda)
set(CMAKE_CUDA_STANDARD 11)

# MACROS glm
add_definitions(-DGLM_SWIZZLE)
add_definitions(-DGLM_FORCE_RADIANS)
add_definitions(-DGLEW_STATIC)
add_definitions(-DGLM_ENABLE_EXPERIMENTAL)

# MACROS eigen
add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_ZERO)

# MACROS feh
add_definitions(-DFEH_RENDER_USE_SEPARATE_ATTRIB_FORMAT)
add_definitions(-DFEH_USE_GPU_ONED_SEARCH)
add_definitions(-DFEH_MULTI_OBJECT_MODEL)
add_definitions(-DFEH_CORE_USE_COLOR_INFO)
#add_definitions(-DFEH_USE_REGION_TRACKER)

# opengl
find_package(OpenGL REQUIRED)
# glfw
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
#find_package(GLEW REQUIRED) # NOT required for the core library, but for annotation tools
# OpenCV
find_package(OpenCV REQUIRED)
# Glog
find_package(Glog REQUIRED)
# Eigen
find_package(Eigen3 REQUIRED)
# Protobuf
find_package(Protobuf REQUIRED)
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${PROJECT_SOURCE_DIR}/protocols/vlslam.proto)

# multi-precision floating point numbers support
# for numerical issues of un-normalized likelihood
find_package(MPFR REQUIRED)
find_package(MPFR_CXX REQUIRED)
find_package(GMP REQUIRED)

# cpu parallelism via intel threading building blocks
find_package(TBB REQUIRED)

# facebook folly
find_package(Folly REQUIRED)


include_directories(
        ${CMAKE_CURRENT_BINARY_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/thirdparty/glad/include
        ${PROJECT_SOURCE_DIR}/thirdparty/Sophus/include
        ${PROJECT_SOURCE_DIR}/thirdparty/googletest/include
        ${PROJECT_SOURCE_DIR}/thirdparty/libigl/include

        ${GLFW_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PROTOBUF_INCLUDE_DIRS}
        ${GMP_INCLUDES}
        ${MPFR_INCLUDES}
        ${MPFR_CXX_INCLUDES}
        ${FOLLY_INCLUDE_DIR}

#        /usr/local/include
)

link_directories(
        ${PROJECT_SOURCE_DIR}/lib
        ${PROJECT_SOURCE_DIR}/thirdparty/googletest/lib

#        /usr/local/lib
#        /usr/lib/x86_64-linux-gnu
)

set(OpenCV_MINIMAL
        opencv_highgui
        opencv_core
        opencv_imgcodecs
        opencv_imgproc
        opencv_videoio
        opencv_calib3d)


link_libraries(
        glfw
        glog
        protobuf
        tbb
        ${OpenCV_MINIMAL}
        ${GMP_LIBRARIES}
        ${MPFR_LIBRARIES}
        ${FOLLY_LIBRARY}
        lcm
        double-conversion   # for folly
        GL
)


add_library(feh SHARED
        src/scene_inference.cpp
        src/scene.cpp
        src/renderer.cpp
        src/renderer_utils.cpp
        src/region_based_tracker.cpp
        src/tracker.cpp
        src/tracker_sir.cpp
        src/tracker_init.cpp
        src/tracker_utils.cpp
        src/dataloaders.cpp
        src/oned_search.cpp
        src/bresenham.cpp
        src/utils.cpp
        src/matdiff.cpp
        ${PROJECT_SOURCE_DIR}/thirdparty/glad/src/glad.c
        ${PROTO_SRCS})


# generate string files for including shader code
#add_custom_command(OUTPUT shader_strings
#        COMMAND python ${PROJECT_SOURCE_DIR}/generate_shader_strings.py
#        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
#add_custom_target(shader_strings_autogen
#        DEPENDS shader_strings)
#add_dependencies(feh shader_strings_autogen)
#
#set_target_properties(feh PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON
#        CUDA_STANDARD 11)

add_executable(sot app/single_object_tracking.cpp)
target_link_libraries(sot feh)

add_executable(mot app/multiple_object_tracking.cpp)
target_link_libraries(mot feh)

add_executable(preprocess_mesh app/preprocess_mesh.cpp)
target_link_libraries(preprocess_mesh feh)

add_executable(linemod app/linemod.cpp)
target_link_libraries(linemod feh)

#add_executable(test_surface_error test/test_surface_error.cpp)
#target_link_libraries(test_surface_error feh)

add_executable(test_kitti test/test_kitti.cpp)
target_link_libraries(test_kitti feh)

add_executable(inspector app/inspector.cpp app/viewer.cpp)
target_link_libraries(inspector feh)

################################################################################
################################################################################
################################################################################
# ROS (optionally build ros-based viewer)
################################################################################
################################################################################
################################################################################
set(build_viewer False)
if (build_viewer)
    find_package(catkin REQUIRED COMPONENTS
            std_msgs
            message_generation
            image_transport
            sensor_msgs
            cv_bridge
            roscpp
            pcl_ros)
    find_package(roscpp REQUIRED)

    catkin_package(CATKIN_DEPENDS
            std_msgs
            message_runtime
            message_generation
            sensor_msgs
            image_transport
            cv_bridge
            roscpp
            CATKIN_DEPENDS message_runtime
            INCLUDE_DIRS
            LIBRARIES)
    include_directories(${catkin_INCLUDE_DIRS})
    add_definitions(${PCL_DEFINITIONS})
    include_directories(
            ${PCL_INCLUDE_DIRS}
    )
    link_libraries(
            ${PCL_LIBRARY_DIRS}
    )

    add_executable(rosviewer app/rosviewer.cpp app/viewer.cpp)
    target_link_libraries(rosviewer
            feh
            ${catkin_LIBRARIES}
            ${roscpp_LIBRARIES}
            ${PCL_COMMON_LIBRARIES})
endif(build_viewer)

#add_executable(test_octree test/test_octree.cpp)

#add_executable(folly_json test/test_folly_json.cpp)

#add_executable(rigidpose app/rigidpose.cpp)
#target_link_libraries(rigidpose feh)

# add_executable(test_render test/test_render.cpp)
# target_link_libraries(test_render feh)

#add_executable(test_onedsearch test/test_onedsearch.cpp)
#target_link_libraries(test_onedsearch feh)

#add_executable(test_differential test/test_differential.cpp)
#target_link_libraries(test_differential feh)

#add_executable(test_objloader test/test_objloader.cpp)
#target_link_libraries(test_objloader feh)

#add_executable(test_protobufloader test/test_protobufloader.cpp)
#target_link_libraries(test_protobufloader feh)

#add_executable(test_tracker test/test_tracker.cpp)
#target_link_libraries(test_tracker feh)
#
#add_executable(test_dt test/test_distance_transform.cpp)
#target_link_libraries(test_dt feh)
#
#add_executable(test_mydt test/test_mydt.cpp)
#target_link_libraries(test_mydt feh)
#
#add_executable(test_particle test/test_particles.cpp)
#target_link_libraries(test_particle feh)
#
#add_executable(test_region test/test_region.cpp)
#target_link_libraries(test_region feh)
#
#add_executable(test_wireframe test/test_wireframe.cpp)
#target_link_libraries(test_wireframe feh)
#
#add_executable(test_multirenderer test/test_multirenderer.cpp)
#target_link_libraries(test_multirenderer feh)

#add_executable(test_delaunay test/test_delaunay.cpp)
#target_link_libraries(test_delaunay feh)

#add_executable(test_ukf test/test_ukf.cpp)
#target_link_libraries(test_ukf feh gtest gtest_main)

#add_executable(test_ukf_mackey_glass test/test_ukf_mackey_glass.cpp)
#target_link_libraries(test_ukf_mackey_glass)



#################################################
# differentiable likelihood for 3D tracking
#################################################
add_executable(pix3dloader src/pix3d/test_loader.cpp)
target_link_libraries(pix3dloader feh)



#################################################
# installation command
#################################################
install(TARGETS feh
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)






