cmake_minimum_required(VERSION 3.15)
project(faceAlignmentAccuracy)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -ffast-math -Ofast -fopenmp")

set(CMAKE_CXX_STANDARD 14)
add_subdirectory(dlib)

include_directories("${CMAKE_CURRENT_LIST_DIR}/lib/ncnn/build_amd64/install/include/ncnn")
include_directories(${CMAKE_CURRENT_LIST_DIR})

set(LIBNCNN ${CMAKE_CURRENT_LIST_DIR}/lib/ncnn/build_amd64/install/lib/libncnn.a)


#add_executable(faceAlignmentAccuracy src/main.cpp src/mtcnn_original/mtcnn.hpp  src/mtcnn_original/mtcnn.cpp)
add_executable(faceAlignmentAccuracy src/main.cpp src/mtcnn_new/mtcnn.h src/mtcnn_new/mtcnn.cpp)
target_link_directories(faceAlignmentAccuracy PUBLIC {CMAKE_CURRENT_LIST_DIR}/lib)
target_include_directories(faceAlignmentAccuracy PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(faceAlignmentAccuracy PUBLIC opencv_core opencv_highgui opencv_video opencv_videoio opencv_imgcodecs opencv_imgproc opencv_calib3d dlib ${LIBNCNN} stdc++fs)
