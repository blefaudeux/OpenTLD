cmake_minimum_required(VERSION 2.6)

include_directories(${OpenCV_INCLUDE_DIRS})
add_library(cvblobs blob.cpp BlobContour.cpp BlobOperators.cpp BlobProperties.cpp BlobResult.cpp ComponentLabeling.cpp)
