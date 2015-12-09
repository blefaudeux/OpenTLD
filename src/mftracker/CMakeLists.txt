cmake_minimum_required(VERSION 2.6)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(mftracker
    bb.cpp
    bb_predict.cpp
    fbtrack.cpp
    lk.cpp
    median.cpp)
    
target_link_libraries(mftracker ${OpenCV_LIBS})
