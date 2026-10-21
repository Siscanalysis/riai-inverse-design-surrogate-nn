cmake_minimum_required(VERSION 3.24)

# Probe build: the same drivers against the NEWER OpenNN (C:/rai_onn_new/src, commit 1e4eceb0d
# plus the measurement instrument), kept entirely apart from the paper's pinned build so that
# code/bin and C:/rai_onn/build stay exactly as the published numbers were produced.
project(rai_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OPENNN_SOURCE_DIR "C:/rai_onn_new/src" CACHE PATH "Newer OpenNN snapshot")

set(OpenNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(OpenNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(OpenNN_BUILD_VISION OFF CACHE BOOL "" FORCE)
set(OpenNN_ENABLE_LTO OFF CACHE BOOL "" FORCE)
set(OpenNN_DISABLE_CUDA ON CACHE BOOL "" FORCE)
set(OpenNN_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory(${OPENNN_SOURCE_DIR} opennn EXCLUDE_FROM_ALL)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

add_executable(probe_sample probe_sample.cpp)
target_include_directories(probe_sample PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(probe_sample PRIVATE opennn)
