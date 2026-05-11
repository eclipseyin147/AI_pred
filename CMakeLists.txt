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
target_include_directories(tju-torch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${TORCH_INCLUDE_DIRS}")

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

# Install dependency DLLs on Windows
if (WIN32)
    set(VCPKG_BIN_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")

    # --- PyTorch Core DLLs ---
    set(TORCH_CORE_DLLS
            c10.dll
            torch_cpu.dll
            torch_cuda.dll
            c10_cuda.dll
    )

    # --- PyTorch Indirect Dependencies ---
    set(TORCH_DEP_DLLS
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
            sleef.dll
            magma.dll
    )

    # --- matio & HDF5 Dependencies ---
    set(MATIO_DLLS
            libmatio.dll
            hdf5.dll
            szip.dll
            zlib1.dll
    )

    # --- MinGW Runtime (required by openblas / liblapack / libgfortran) ---
    set(MINGW_DLLS
            libgcc_s_seh-1.dll
            libwinpthread-1.dll
    )

    set(ALL_DEP_DLLS)
    foreach (DLL_LIST IN ITEMS TORCH_CORE_DLLS TORCH_DEP_DLLS MATIO_DLLS MINGW_DLLS)
        foreach (DLL ${${DLL_LIST}})
            set(DLL_PATH "${VCPKG_BIN_DIR}/${DLL}")
            if (EXISTS "${DLL_PATH}")
                list(APPEND ALL_DEP_DLLS "${DLL_PATH}")
            else ()
                message(WARNING "Dependency DLL not found in vcpkg: ${DLL}")
            endif ()
        endforeach ()
    endforeach ()

    # --- CUDA Runtime DLLs (auto-detect from CUDA_PATH) ---
    if (DEFINED ENV{CUDA_PATH})
        set(CUDA_BIN_DIR "$ENV{CUDA_PATH}/bin")
        set(CUDA_RUNTIME_DLLS
                cublas64_12.dll
                cublasLt64_12.dll
                cufft64_11.dll
                cusolver64_11.dll
                cusparse64_12.dll
                cudart64_12.dll
        )
        foreach (DLL ${CUDA_RUNTIME_DLLS})
            set(DLL_PATH "${CUDA_BIN_DIR}/${DLL}")
            if (EXISTS "${DLL_PATH}")
                list(APPEND ALL_DEP_DLLS "${DLL_PATH}")
            else ()
                message(WARNING "CUDA runtime DLL not found: ${DLL_PATH}")
            endif ()
        endforeach ()
    else ()
        message(STATUS "CUDA_PATH not set; CUDA runtime DLLs will not be installed.")
    endif ()

    if (ALL_DEP_DLLS)
        install(FILES ${ALL_DEP_DLLS} DESTINATION bin)
    endif ()
endif ()

# Build info output
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
