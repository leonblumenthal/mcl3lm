cmake_minimum_required(VERSION 3.23)

project(mcl3lm)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)



find_package(Eigen3 REQUIRED)
find_package(ceres REQUIRED)
find_package(Sophus REQUIRED)
add_library(Sophus INTERFACE IMPORTED)
find_package(OpenCV REQUIRED)
add_library(OpenCV INTERFACE IMPORTED)
set_property(TARGET OpenCV PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${OpenCV_INCLUDE_DIRS})
set_property(TARGET OpenCV PROPERTY INTERFACE_LINK_LIBRARIES ${OpenCV_LIBS})

find_package(tbb REQUIRED)
add_library(tbb INTERFACE IMPORTED)
set_property(TARGET tbb PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${tbb_INCLUDE_DIRS})
set_property(TARGET tbb PROPERTY INTERFACE_LINK_LIBRARIES ${tbb_LIBS})

set(CMAKE_FIND_DEBUG_MODE TRUE)
find_package(opengv REQUIRED)

find_package(fmt REQUIRED)
find_package(glog REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(evaluation evaluation.cpp pcl_map.h features.h visual_odometry_utils.h io_utils.h visualization_utils.h camera.h visual_odometry.h reports.h alignment.h)
target_link_libraries(evaluation Eigen3::Eigen ceres OpenCV Sophus::Sophus opengv fmt::fmt nlohmann_json::nlohmann_json)