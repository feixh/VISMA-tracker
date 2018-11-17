cmake_minimum_required(VERSION 3.5)

project(vlslam_plus)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMakeModules)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall -g -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native -msse -msse2 -msse3 -msse4")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare -Wno-unused-function -Wno-unused-variable -Wno-narrowing")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin )
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib )

# MACROS glm
add_definitions(-DGLM_SWIZZLE)
add_definitions(-DGLM_FORCE_RADIANS)
add_definitions(-DGLEW_STATIC)
add_definitions(-DGLM_ENABLE_EXPERIMENTAL)

# MACROS eigen
add_definitions(-DEIGEN_INITIALIZE_MATRICES_BY_ZERO)
add_definitions(-DEIGEN_DEFAULT_TO_ROW_MAJOR)

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

add_subdirectory(thirdparty/abseil-cpp)
add_subdirectory(thirdparty/jsoncpp)

include_directories(
        ${CMAKE_CURRENT_BINARY_DIR}
        ${PROJECT_SOURCE_DIR}/core
        ${PROJECT_SOURCE_DIR}/tracker
        ${PROJECT_SOURCE_DIR}/pix3d

        ${PROJECT_SOURCE_DIR}/thirdparty/glad/include
        ${PROJECT_SOURCE_DIR}/thirdparty/googletest/include
        ${PROJECT_SOURCE_DIR}/thirdparty/libigl/include
        ${PROJECT_SOURCE_DIR}/thirdparty/lcm-1.4.0/include
        ${PROJECT_SOURCE_DIR}/thirdparty/jsoncpp/include
        ${PROJECT_SOURCE_DIR}/thirdparty/abseil-cpp
        ${PROJECT_SOURCE_DIR}/thirdparty/zmqpp/include

        ${GLFW_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PROTOBUF_INCLUDE_DIR}
        ${GMP_INCLUDES}
        ${MPFR_INCLUDES}
        ${MPFR_CXX_INCLUDES}
        /usr/include
)

link_directories(
        ${PROJECT_SOURCE_DIR}/lib
        ${PROJECT_SOURCE_DIR}/thirdparty/googletest/lib
        ${PROJECT_SOURCE_DIR}/thirdparty/lcm-1.4.0/lib
        # ${PROJECT_SOURCE_DIR}/thirdparty/jsoncpp/lib
        ${PROJECT_SOURCE_DIR}/thirdparty/zmqpp/lib
        /usr/lib/x86_64-linux-gnu/
)

link_libraries(
        glfw
        glog
        tbb

        opencv_highgui
        opencv_core
        opencv_imgproc
        opencv_video
        opencv_calib3d

        ${GMP_LIBRARIES}
        ${MPFR_LIBRARIES}
        ${PROTOBUF_LIBRARY}
        lcm
        GL
        gtest gtest_main    # for testing
        gflags
        zmq zmqpp
)

add_library(core SHARED
        core/utils.cpp)

add_library(feh SHARED
        tracker/scene_inference.cpp
        tracker/scene.cpp
        tracker/renderer.cpp
        tracker/region_based_tracker.cpp
        tracker/tracker.cpp
        tracker/tracker_sir.cpp
        tracker/tracker_init.cpp
        tracker/tracker_utils.cpp
        tracker/dataloaders.cpp
        tracker/oned_search.cpp
        tracker/bresenham.cpp
        tracker/DFtracker.cpp
        tracker/gravity_aligned_tracker.cpp
        ${PROJECT_SOURCE_DIR}/thirdparty/glad/src/glad.c
        ${PROTO_SRCS})

## generate string files for including shader code
#add_custom_command(
#        TARGET feh
#        COMMAND python installation_scripts/generate_shader_strings.py
#        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

# Once feh is built, link applications to it.
link_libraries(feh core absl::str_format absl::time jsoncpp)

add_executable(preprocess_mesh app/preprocess_mesh.cpp)
add_executable(mot app/MOT_visma.cpp)
add_executable(sot app/SOT_visma.cpp)
add_executable(sorbt_linemod app/SORBT_linemod.cpp)
add_executable(sorbt_rigidpose app/SORBT_rigidpose.cpp)
add_executable(sodft_visma app/SODFT_visma.cpp)


#################################################
# differentiable likelihood for 3D tracking
#################################################
add_executable(pix3dloader pix3d/test_loader.cpp)
add_executable(pix3dkeypoints pix3d/generate_control_points.cpp)
add_executable(pix3dtracker pix3d/test_tracker.cpp pix3d/diff_tracker.cpp)

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
            ${catkin_LIBRARIES}
            ${roscpp_LIBRARIES}
            ${PCL_COMMON_LIBRARIES})
endif(build_viewer)

#add_executable(test_octree test/test_octree.cpp)
#add_executable(folly_json test/test_folly_json.cpp)
#add_executable(test_render test/test_render.cpp)
#add_executable(test_onedsearch test/test_onedsearch.cpp)
#add_executable(test_differential test/test_differential.cpp)
#add_executable(test_objloader test/test_objloader.cpp)
#add_executable(test_protobufloader test/test_protobufloader.cpp)
#add_executable(test_tracker test/test_tracker.cpp)
#add_executable(test_dt test/test_distance_transform.cpp)
#add_executable(test_mydt test/test_mydt.cpp)
#add_executable(test_particle test/test_particles.cpp)
#add_executable(test_region test/test_region.cpp)
#add_executable(test_wireframe test/test_wireframe.cpp)
#add_executable(test_multirenderer test/test_multirenderer.cpp)
#add_executable(test_delaunay test/test_delaunay.cpp)
#add_executable(test_ukf test/test_ukf.cpp)
#add_executable(test_ukf_mackey_glass test/test_ukf_mackey_glass.cpp)
#target_link_libraries(test_ukf_mackey_glass)


#################################################
# installation command
#################################################
install(TARGETS feh
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
