cmake_minimum_required(VERSION 3.16)
project(colss LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Prefer modern FindPython behavior for pybind11 (avoid CMP0148 warnings)
# This must be set BEFORE find_package(pybind11).
set(PYBIND11_FINDPYTHON ON CACHE BOOL "Let pybind11 use FindPython/FindPython3")

# Find pybind11
find_package(pybind11 REQUIRED)

# Better OpenMP handling for macOS
if(APPLE)
    # Check if we're using AppleClang
    if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        # For AppleClang, use Homebrew libomp flags if available
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "omp")
        find_library(OpenMP_omp_LIBRARY
            NAMES omp
            HINTS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib
            PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib
            NO_DEFAULT_PATH
        )
        if(OpenMP_omp_LIBRARY)
            message(STATUS "Found OpenMP library: ${OpenMP_omp_LIBRARY}")
            set(OpenMP_CXX_LIBRARIES ${OpenMP_omp_LIBRARY})
            find_path(OpenMP_CXX_INCLUDE_DIR
                NAMES omp.h
                HINTS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include
                PATHS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include
                NO_DEFAULT_PATH
            )
            if(OpenMP_CXX_INCLUDE_DIR)
                set(OpenMP_CXX_INCLUDE_DIRS ${OpenMP_CXX_INCLUDE_DIR})
            endif()
        endif()
    endif()
endif()

# Windows/MSVC: use experimental OpenMP SIMD flag if desired
if(MSVC)
    # Add compile option for MSVC (applied to targets later)
    add_compile_options(/openmp:experimental)
endif()

# Find OpenMP in a portable way
find_package(OpenMP REQUIRED)

# Build pybind11 module
pybind11_add_module(_colss binding.cpp)

# Link OpenMP (preferred target)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_colss PRIVATE OpenMP::OpenMP_CXX)
elseif(APPLE AND OpenMP_omp_LIBRARY)
    # Fallback for older CMake / tricky macOS setups
    target_compile_options(_colss PRIVATE -Xpreprocessor -fopenmp)
    target_include_directories(_colss PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
    target_link_libraries(_colss PRIVATE ${OpenMP_omp_LIBRARY})
endif()

# Ensure MSVC OpenMP flags apply to the target
if(MSVC)
    target_compile_options(_colss PRIVATE /openmp:experimental)
endif()

# Vectorized Math Options (xsimd, SLEEF, Intel SVML)
option(USE_XSIMD "Use xsimd for vectorized math" OFF)
option(USE_SLEEF "Use SLEEF for vectorized math" OFF)
option(USE_SVML "Use Intel SVML for vectorized math" OFF)

if(USE_XSIMD)
    target_compile_definitions(_colss PRIVATE USE_XSIMD)
    find_package(xsimd QUIET CONFIG)
    if(xsimd_FOUND)
        target_link_libraries(_colss PRIVATE xsimd)
    endif()
endif()

if(USE_SLEEF)
    target_compile_definitions(_colss PRIVATE USE_SLEEF)
    find_library(SLEEF_LIBRARY NAMES sleef)
    if(SLEEF_LIBRARY)
        target_link_libraries(_colss PRIVATE ${SLEEF_LIBRARY})
    else()
        target_link_libraries(_colss PRIVATE sleef)
    endif()
endif()

if(USE_SVML)
    target_compile_definitions(_colss PRIVATE USE_SVML)
    find_library(SVML_LIBRARY NAMES svml)
    if(SVML_LIBRARY)
        target_link_libraries(_colss PRIVATE ${SVML_LIBRARY})
    else()
        target_link_libraries(_colss PRIVATE svml)
    endif()
endif()

# Project include dir
target_include_directories(_colss PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Compiler-specific optimizations
if(MSVC)
    target_compile_options(_colss PRIVATE /O2)
else()
    target_compile_options(_colss PRIVATE -O3 -march=native -mtune=native -ffast-math -fno-finite-math-only -funroll-loops)
endif()

# Install targets
install(TARGETS _colss DESTINATION colss)
install(FILES ${CMAKE_SOURCE_DIR}/colss/__init__.py DESTINATION colss)
