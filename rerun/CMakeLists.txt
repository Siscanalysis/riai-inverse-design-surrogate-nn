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

# OpenNN dependency, PINNED for reproducibility to commit 5598487e3 (branch dev).
# The public headers at that commit are VENDORED under _vendor/opennn/ so the build is
# immune to the ongoing refactor of the live source tree. We still link the prebuilt
# static library (built from the same commit) and reuse its own Eigen copy (to avoid the
# two-Eigen COMDAT mix). Override the build dir if your matching-commit build lives
# elsewhere:  cmake -S . -B build -DOPENNN_BUILD=/path/to/opennn/build-ninja
set(OPENNN_BUILD "C:/Users/Artelnics/Documents/opennn/build-ninja" CACHE PATH "OpenNN build dir with the commit-5598487e3 static lib")
set(OPENNN_VENDOR "${CMAKE_CURRENT_SOURCE_DIR}/_vendor")

find_package(OpenMP)

set(_riai_common_includes
    ${OPENNN_VENDOR}                 # pinned "opennn/*.h" headers (commit 5598487e3)
    ${OPENNN_VENDOR}/opennn
    ${OPENNN_BUILD}                  # build-generated headers matching the lib
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
riai_add(baselines  baselines.cpp)
