cmake_minimum_required(VERSION 3.16)

project(image_compression
    VERSION 1.0
    LANGUAGES CXX
)

# ============================================================
# C++ configuration
# ============================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# Dependencies
# ============================================================

find_package(OpenCV REQUIRED)

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
    src/Quantizers/VectorQuantizer.cpp
)

# ============================================================
# Include directories
# ============================================================

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# ============================================================
# Libraries
# ============================================================

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
)

# ============================================================
# Compiler warnings
# ============================================================

if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE
        /W4
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()