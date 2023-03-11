cmake_minimum_required(VERSION 2.8.3)
project(msckf)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)
find_package(OpenCV REQUIRED)

if(OPENCV_FOUND) 
message(STATUS "The Opecv lib is found!") 
endif()

message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${CERES_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_library(msckf_vio SHARED
	src/System.cpp
  src/parameters.cpp
  src/image_processor.cpp
  src/msckf_vio.cpp
  # src/utils.cpp
)
target_link_libraries(msckf_vio
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  ${Pangolin_LIBRARIES}
)
add_executable(run_test test/run_test.cpp)
target_link_libraries(run_test msckf_vio)


