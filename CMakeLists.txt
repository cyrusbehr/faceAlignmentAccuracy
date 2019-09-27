cmake_minimum_required(VERSION 3.15)
project(faceAlignmentAccuracy)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -ffast-math -Ofast -fopenmp")

set(CMAKE_CXX_STANDARD 14)
add_subdirectory(dlib)

add_executable(faceAlignmentAccuracy main.cpp)
target_link_directories(faceAlignmentAccuracy PUBLIC {CMAKE_CURRENT_LIST_DIR}/lib)
target_include_directories(faceAlignmentAccuracy PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(faceAlignmentAccuracy PUBLIC opencv_core opencv_highgui opencv_video opencv_imgcodecs opencv_imgproc opencv_calib3d dlib stdc++fs)
