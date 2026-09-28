cmake_minimum_required(VERSION 3.16)

project(image_compression
    VERSION 1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# OpenCV
# ============================================================

include(FetchContent)

# Only build the OpenCV modules used by this project.
set(BUILD_LIST "core,imgproc,imgcodecs,highgui" CACHE STRING "" FORCE)

# Disable unnecessary OpenCV components.
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "" FORCE)

# Disable language bindings.
set(BUILD_opencv_python2 OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_java OFF CACHE BOOL "" FORCE)

# Disable hardware/media backends we don't need.
set(WITH_IPP OFF CACHE BOOL "" FORCE)
set(WITH_OPENCL OFF CACHE BOOL "" FORCE)
set(WITH_CUDA OFF CACHE BOOL "" FORCE)
set(WITH_FFMPEG OFF CACHE BOOL "" FORCE)
set(WITH_GSTREAMER OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG 4.12.0
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(opencv)

# ============================================================
# Application
# ============================================================

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/Images/Image.cpp
    src/Images/ImageGray.cpp
    src/Images/ImageBGR.cpp
    src/Quantizers/BTCQuantizer.cpp
    src/Quantizers/DCTQuantizer.cpp
    src/Quantizers/DPCMQuantizer.cpp
    src/Quantizers/ScalarQuantizer.cpp
    src/Quantizers/VectorQuantizer.cpp
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${opencv_SOURCE_DIR}/include
        ${opencv_SOURCE_DIR}/modules/core/include
        ${opencv_SOURCE_DIR}/modules/imgproc/include
        ${opencv_SOURCE_DIR}/modules/imgcodecs/include
        ${opencv_SOURCE_DIR}/modules/highgui/include
        ${opencv_BINARY_DIR}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_highgui
)

# ============================================================
# Compiler warnings
# ============================================================

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()