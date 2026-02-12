cmake_minimum_required(VERSION 3.16)
project(colss LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

pybind11_add_module(colss binding.cpp)
target_link_libraries(colss PRIVATE OpenMP::OpenMP_CXX)

install(TARGETS colss DESTINATION .)
