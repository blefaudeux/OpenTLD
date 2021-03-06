# Open The CMake GUI
# specify the source directory and the binary directory
# press configure
# set CMAKE_INSTALL_PREFIX to the path where you want to install the program
# press configure
# check BUILD_WITH_QT if you want to build the program with a QT-Config GUI
# check GENERATE_DEB_PACKAGE if you want to build a debian package (only on Linux)
#
# UNIX Makefile:
# 1) go to the binary folder and type "make" to build the project
# 2) (optional) type "make install all" to install the files into the install
#    directory
# 3) (optional) type "make package" to create a package of the install folder
#    (.tgz file if GENERATE_DEB_PACKAGE=false, .deb file if GENERATE_DEB_PACKAGE=true)
#
# Microsoft Visual C++:
# 1) open the .sln file
# 2) change the mode to "Release" -> only Release is supported!
# 3) build the project "ALL_BUILD" to build the tld project
# 4) build the project "INSTALL" to install the files into the install
#    directory
# 5) build the project "PACKAGE" to create an NSIS-installer (NSIS is required)


project(OpenTLD)

cmake_minimum_required(VERSION 2.6)

# Enable C++11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#-------------------------------------------------------------------------------
#version
set(TLD_VERSION_MAJOR 1)
set(TLD_VERSION_MINOR 2)
set(TLD_VERSION_PATCH 1)
set(TLD_VERSION ${TLD_VERSION_MAJOR}.${TLD_VERSION_MINOR}.${TLD_VERSION_PATCH})

#------------------------------------------------------------------------------
#build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)

if(WIN32)
	add_definitions(-DLIBCONFIGXX_STATIC -DLIBCONFIG_STATIC) #Needed when linking libconfig statically
endif(WIN32)

if(APPLE)
	add_definitions(-DHAVE_XLOCALE_H)
endif(APPLE)

if(WIN32)
	find_path(OpenCV_DIR OpenCVConfig.cmake PATHS 
			  "C:/OpenCV/opencv-2.4.9/build" # PC Amaury Mines
			  # add your OpenCV path here : 
			  )
	if( ${OpenCV_DIR} STREQUAL "OpenCV_DIR-NOTFOUND" ) 
		message( FATAL_ERROR "FATAL_ERROR : OpenCV is not found, Please insert your OpenCV Path in pl_utils.cmake")
	else ()
		message(STATUS "Your OpenCV_DIR is ${OpenCV_DIR}")
	endif()
	find_package(OpenCV REQUIRED)
else()
	find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgomp")

include_directories(src/cvblobs src/imacq src/config src/tld src/mftracker src/main ${OpenCV_INCLUDE_DIRS})

# Build Halide
set(HALIDE halide)
add_custom_target(
    halide
    cd ${CMAKE_SOURCE_DIR}/dependencies/Halide && make
)

#-------------------------------------------------------------------------------
#add subdirectories
add_subdirectory(src/imacq)
add_subdirectory(src/cvblobs)
add_subdirectory(src/config)
add_subdirectory(src/tld)
add_subdirectory(src/mftracker)
add_subdirectory(src/main)
add_subdirectory(src)

add_dependencies(opentld ${HALIDE})

configure_file("${PROJECT_SOURCE_DIR}/OpenTLDConfig.cmake.in" "${PROJECT_BINARY_DIR}/OpenTLDConfig.cmake" @ONLY)

