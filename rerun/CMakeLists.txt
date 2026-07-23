cmake_minimum_required(VERSION 3.18)
project(riai_inverse LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

if(MSVC)
    add_compile_options(/arch:AVX2 /bigobj)
    add_compile_definitions(NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

# OpenNN dependency. Pinned to the stable snapshot the experiments were built and
# tested against: OpenNN commit 5598487e3 (branch dev), prebuilt under build-ninja,
# with its OWN Eigen copy to avoid the two-Eigen COMDAT mix. Override both paths on
# the command line if your checkout/build live elsewhere:
#   cmake -S . -B build -DOPENNN_ROOT=/path/to/opennn -DOPENNN_BUILD=/path/to/build
set(OPENNN_ROOT  "C:/Users/Artelnics/Documents/opennn" CACHE PATH "OpenNN source root")
set(OPENNN_BUILD "${OPENNN_ROOT}/build-ninja"          CACHE PATH "OpenNN build dir (prebuilt lib)")

find_package(OpenMP)

set(_riai_common_includes
    ${OPENNN_ROOT}
    ${OPENNN_ROOT}/opennn
    ${OPENNN_BUILD}
    ${OPENNN_BUILD}/_deps/eigen-src)

set(_riai_common_libs
    ${OPENNN_BUILD}/opennn/opennn.lib
    ${OPENNN_BUILD}/_deps/opennn_zlib-build/zs.lib
    ${OPENNN_BUILD}/_deps/opennn_libjpeg_turbo-install/lib/jpeg-static.lib)

function(riai_add name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${_riai_common_includes})
    target_link_libraries(${name} PRIVATE ${_riai_common_libs})
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

riai_add(case_a_cnc case_a_cnc.cpp)
riai_add(case_b_hea case_b_hea.cpp)
