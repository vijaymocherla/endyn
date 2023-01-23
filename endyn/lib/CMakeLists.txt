if (POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW)
endif ()

cmake_policy(SET CMP0048 NEW)  # project_VERSION* variables populated from project(... VERSION x.x.x) string

project(endyn
VERSION 0.1.0
LANGUAGES CXX)
set(endyn_AUTHORS      "Sai Vijay Mocherla")
set(endyn_DESCRIPTION  "Ab initio package for electron-nuclear dynamics")
set(endyn_URL          "https://github.com/vijaymocherla/endyn")
set(endyn_LICENSE      "GPL-3.0+")
include_directories(integrators)
include_directories(bindings)
aux_source_directory(integrators SRC_LIST)
aux_source_directory(bindings SRC_LIST)

# currently forcing cmake to use icpx with mkl
set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_FC_COMPILER ifort)
set(OpenMP_FIND_COMPONENTS "CXX")
# find_package(psi4 1.4 REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(MKL REQUIRED)
if(MKL_FOUND)
    set(EIGEN_USE_MKL_ALL 1) # This will go into config.h - it makes Eigen use MKL
    include_directories(${MKL_INCLUDE_DIR})
else()
    set(EIGEN_USE_MKL_ALL 0)
endif()

set(CMAKE_BINARY_DIR "${CMAKE_BINARY_DIR}; ${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")
# find_package(LAPACK REQUIRED)
# find_package(LAPACKE REQUIRED)
# find_package(BLAS REQUIRED)

find_package(fmt REQUIRED)
find_package(Python 3.9 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Including custom cmake rules
include(cmake/clang-cxx-dev-tools.cmake)
include(autocmake_omp) 

add_compile_options(-Wall -Wextra -pedantic) # adding some compiler flags
if(CODE_COVERAGE)
  message("-- Code coverage enabled")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
endif(CODE_COVERAGE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++1z-extensions") # avoid warnings for C++17

pybind11_add_module(_endyn ${SRC_LIST} ${module_SOURCES})
