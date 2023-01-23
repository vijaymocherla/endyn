cmake_minimum_required(VERSION 3.15...3.22)

project(endyn)

set(CMAKE_CXX_STANDARD 17)

option(CODE_COVERAGE "Enable coverage reporting" OFF)

add_subdirectory(endyn/lib)