# CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)

project(tju-torch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED CONFIG)
find_package(OpenMP REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

# Find matio library
find_package(matio CONFIG REQUIRED)
if(NOT matio_FOUND)
    # Fallback: try to find it manually
    find_library(MATIO_LIBRARY NAMES matio)
    find_path(MATIO_INCLUDE_DIR matio.h)
    if(MATIO_LIBRARY AND MATIO_INCLUDE_DIR)
        set(MATIO_FOUND TRUE)
        message(STATUS "Found matio: ${MATIO_LIBRARY}")
    else()
        message(FATAL_ERROR "matio library not found")
    endif()
endif()

# Unified executable replacing FFNPredictor, SEDM, and faultDiag
add_executable(tju-torch
    unified_main.cpp
    ffn_manager.cpp
    sedm_manager.cpp
    faultdiag_manager.cpp
    faultDiagnosis.cpp
    training_controller.cpp
)

target_link_libraries(tju-torch "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)
target_include_directories(tju-torch PRIVATE faultDiagnosis.h "${TORCH_INCLUDE_DIRS}")

# Link matio
if(TARGET matio::matio)
    target_link_libraries(tju-torch MATIO::HDF5 MATIO::ZLIB matio::matio)
elseif(MATIO_LIBRARY)
    target_link_libraries(tju-torch ${MATIO_LIBRARY})
    target_include_directories(tju-torch PRIVATE ${MATIO_INCLUDE_DIR})
endif()

if (OpenMP_CXX_FOUND)
    target_link_libraries(tju-torch OpenMP::OpenMP_CXX)
endif ()

if (MSVC)
    # Windows platform compile options
    target_compile_options(tju-torch PRIVATE /W3 /wd4702)
else ()
    # Linux/Mac platform compile options
    target_compile_options(tju-torch PRIVATE -Wall -Wextra -Wpedantic)
endif ()

# Install rules
install(TARGETS tju-torch DESTINATION bin)

# Install Torch DLLs on Windows (vcpkg)
if (WIN32)
    set(TORCH_DLLS
            c10.dll
            torch_cpu.dll
            glog.dll
            gflags.dll
            openblas.dll
            liblapack.dll
            libprotobuf.dll
            mimalloc.dll
            mimalloc-redirect.dll
            libgfortran-5.dll
            abseil_dll.dll
            libquadmath-0.dll
    )
    foreach (DLL ${TORCH_DLLS})
        file(GLOB DLL_FILE "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/${DLL}")
        if (DLL_FILE)
            list(APPEND ALL_DLLS ${DLL_FILE})
        endif ()
    endforeach ()

    if (ALL_DLLS)
        install(FILES ${ALL_DLLS} DESTINATION bin)
    endif ()
endif ()

# Build info output
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
