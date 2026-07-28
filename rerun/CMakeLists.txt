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

# OpenNN dependency, fully VENDORED for reproducibility and to be immune to the ongoing
# refactor of the live source tree. _vendor/ holds a consistent snapshot:
#   _vendor/opennn/  public headers (committed; small)
#   _vendor/eigen/   Eigen headers (gitignored; regenerable)
#   _vendor/lib/     opennn.lib + zs.lib + jpeg-static.lib (gitignored; ~190 MB)
# The snapshot corresponds to OpenNN commit 5598487e3 (the commit these experiments were
# written and validated against). To regenerate on another machine: build OpenNN at that
# commit (e.g. via `git worktree add <dir> 5598487e3`) and copy opennn.lib + the *.h headers
# + Eigen into _vendor (see README):
set(OPENNN_VENDOR "${CMAKE_CURRENT_SOURCE_DIR}/_vendor" CACHE PATH "Vendored OpenNN snapshot")

find_package(OpenMP)

set(_riai_common_includes
    ${OPENNN_VENDOR}                 # resolves "opennn/*.h"
    ${OPENNN_VENDOR}/opennn
    ${OPENNN_VENDOR}/eigen)          # vendored Eigen (single copy, no two-Eigen mix)

set(_riai_common_libs
    ${OPENNN_VENDOR}/lib/opennn.lib
    ${OPENNN_VENDOR}/lib/zs.lib
    ${OPENNN_VENDOR}/lib/jpeg-static.lib)

function(riai_add name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${_riai_common_includes})
    target_link_libraries(${name} PRIVATE ${_riai_common_libs})
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

riai_add(case_a_cnc  case_a_cnc.cpp)
riai_add(case_b_hea  case_b_hea.cpp)
riai_add(case_c_steel case_c_steel.cpp)
riai_add(train_steel train_steel.cpp)
riai_add(baselines   baselines.cpp)
