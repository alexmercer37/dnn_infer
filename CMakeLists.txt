cmake_minimum_required(VERSION 3.16)
include_directories(.)
project(v1)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(k4a REQUIRED)

include_directories( ${CMAKE_CURRENT_LIST_DIR} ) 
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(include)

set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)   
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CUDA_GEN_CODE "-gencode=arch=compute_75,code=sm_75")
 
set(CUDA_DIR "/usr/local/cuda")
set(CUDNN_DIR    "/usr/local/cuda")



include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_DIR}/include
    ${CUDNN_DIR}/include
    )
link_directories(
    ${OpenCV_LIBS_DIRS}
    ${CUDA_DIR}/lib64
    ${CUDNN_DIR}/lib
    )

add_executable(v1 src/main.cpp src/dnn.cpp)

target_link_libraries(v1 ${OpenCV_LIBS} k4a::k4a)
target_link_libraries(v1  ${OpenCV_LIBS})
