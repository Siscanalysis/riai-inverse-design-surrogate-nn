cmake_minimum_required(VERSION 3.24)

project(rai_inverse_design LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The OpenNN sources the experiments are pinned to (see opennn_pin/README.md).
set(OPENNN_SOURCE_DIR "C:/rai_onn/src" CACHE PATH "Pinned OpenNN snapshot")

set(OpenNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(OpenNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(OpenNN_BUILD_VISION OFF CACHE BOOL "" FORCE)
set(OpenNN_ENABLE_LTO OFF CACHE BOOL "" FORCE)
set(OpenNN_DISABLE_CUDA ON CACHE BOOL "" FORCE)
set(OpenNN_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory(${OPENNN_SOURCE_DIR} opennn EXCLUDE_FROM_ALL)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

foreach(driver train_surrogates inverse_cases eval_network baselines)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/${driver}.cpp")
        add_executable(${driver} src/${driver}.cpp)
        target_include_directories(${driver} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(${driver} PRIVATE opennn)
    endif()
endforeach()
