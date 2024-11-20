cmake_minimum_required(VERSION 3.29)
project(ada_2024_project_outliars)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native -mtune=native -fopenmp")

add_executable(graph_stats src/graph_stats.cxx)
