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

add_executable(FFNPredictor prediction_model_FFN.cpp)
add_executable(SEDM predictionSEDM.cpp)

add_executable(faultDiag faultDiagMain.cpp faultDiagnosis.cpp)

target_link_libraries(FFNPredictor "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)
target_include_directories(FFNPredictor PRIVATE "${TORCH_INCLUDE_DIRS}")

target_link_libraries(SEDM "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)
target_include_directories(SEDM PRIVATE "${TORCH_INCLUDE_DIRS}")

target_link_libraries(faultDiag "${TORCH_LIBRARIES}" nlohmann_json::nlohmann_json)
target_include_directories(faultDiag PRIVATE faultDiagnosis.h "${TORCH_INCLUDE_DIRS}")

# Link matio
if(TARGET matio::matio)
    target_link_libraries(faultDiag MATIO::HDF5 MATIO::ZLIB matio::matio)
elseif(MATIO_LIBRARY)
    target_link_libraries(faultDiag ${MATIO_LIBRARY})
    target_include_directories(faultDiag PRIVATE ${MATIO_INCLUDE_DIR})
endif()

if (OpenMP_CXX_FOUND)
    target_link_libraries(FFNPredictor OpenMP::OpenMP_CXX)
    target_link_libraries(SEDM OpenMP::OpenMP_CXX)
endif ()


if (MSVC)
    # Windows 平台编译选项 - 移除 /WX 以避免将警告视为错误
    target_compile_options(FFNPredictor PRIVATE /W3 /wd4702)
    target_compile_options(SEDM PRIVATE /W3 /wd4702)
    target_compile_options(faultDiag PRIVATE /W3 /wd4702)
else ()
    # Linux/Mac 平台编译选项
    target_compile_options(FFNPredictor PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(SEDM PRIVATE -Wall -Wextra -Wpedantic)
endif ()

# 如果需要，可以添加安装规则
install(TARGETS FFNPredictor DESTINATION bin)
install(TARGETS SEDM DESTINATION bin)
install(TARGETS faultDiag DESTINATION bin)
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

# 添加构建信息输出
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")