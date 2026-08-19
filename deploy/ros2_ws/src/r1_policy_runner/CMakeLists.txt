cmake_minimum_required(VERSION 3.16)
project(r1_policy_runner CXX)

if(NOT CMAKE_BUILD_TYPE)
  # This is a hard-real-time control node; an unoptimised default build would
  # give latency numbers that mean nothing.
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# ---------------------------------------------------------------------------
# TensorRT and the CUDA runtime.
#
# Two supported provenances, and the hints cover both without a branch:
#   * Jetson / JetPack -- everything under /usr, aarch64 multiarch dirs
#   * this dev box     -- pip wheels, surfaced by deploy/setup.sh as link farms
#     under deploy/third_party and exported by deploy/env.sh
#
# CUDAToolkit is deliberately NOT used to find cudart: on this host the apt CUDA
# is 11.5 while TensorRT 10.3's x86 build links libcudart.so.12, and pulling two
# cudart majors into one process is undefined behaviour.
# ---------------------------------------------------------------------------
set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT install prefix")
set(CUDART_ROOT   "$ENV{CUDART_ROOT}"   CACHE PATH "CUDA runtime prefix")

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT}/include /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu /usr/include)
find_path(TENSORRT_ONNX_INCLUDE_DIR NvOnnxParser.h
  HINTS ${TENSORRT_ROOT}/include /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu /usr/include)
find_library(TENSORRT_LIB nvinfer
  HINTS ${TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
find_library(TENSORRT_ONNX_LIB nvonnxparser
  HINTS ${TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)

find_path(CUDART_INCLUDE_DIR cuda_runtime_api.h
  HINTS ${CUDART_ROOT}/include /usr/local/cuda/include /usr/include)
find_library(CUDART_LIB cudart
  HINTS ${CUDART_ROOT}/lib ${CUDART_ROOT}/lib64 /usr/local/cuda/lib64 /usr/lib/aarch64-linux-gnu)

foreach(v TENSORRT_INCLUDE_DIR TENSORRT_ONNX_INCLUDE_DIR TENSORRT_LIB TENSORRT_ONNX_LIB
          CUDART_INCLUDE_DIR CUDART_LIB)
  if(NOT ${v})
    message(FATAL_ERROR
      "${v} not found. On a dev box run deploy/setup.sh then `source deploy/env.sh`; "
      "on a Jetson make sure the JetPack TensorRT packages are installed.")
  endif()
endforeach()

# Report the resolved TensorRT version at configure time. A header/runtime skew
# here is the single most common cause of "it built but the engine won't load".
file(STRINGS ${TENSORRT_INCLUDE_DIR}/NvInferVersion.h _trt_ver_lines REGEX "NV_TENSORRT_(MAJOR|MINOR|PATCH)")
string(REGEX MATCH "NV_TENSORRT_MAJOR ([0-9]+)" _ "${_trt_ver_lines}")
set(_trt_major ${CMAKE_MATCH_1})
string(REGEX MATCH "NV_TENSORRT_MINOR ([0-9]+)" _ "${_trt_ver_lines}")
set(_trt_minor ${CMAKE_MATCH_1})
string(REGEX MATCH "NV_TENSORRT_PATCH ([0-9]+)" _ "${_trt_ver_lines}")
message(STATUS "TensorRT headers ${_trt_major}.${_trt_minor}.${CMAKE_MATCH_1} from ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib     ${TENSORRT_LIB}")
message(STATUS "CUDA runtime     ${CUDART_LIB}")

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
add_library(trt_policy src/trt_policy.cpp)
target_include_directories(trt_policy PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
# SYSTEM: NvInfer.h carries a large number of self-referential deprecation
# warnings (its own INT8 calibrator hierarchy). They are NVIDIA's to fix, and
# burying our own warnings under them defeats -Wall.
target_include_directories(trt_policy SYSTEM PUBLIC
  ${TENSORRT_INCLUDE_DIR} ${TENSORRT_ONNX_INCLUDE_DIR} ${CUDART_INCLUDE_DIR})
target_link_libraries(trt_policy PUBLIC ${TENSORRT_LIB} ${TENSORRT_ONNX_LIB} ${CUDART_LIB})

add_executable(r1_build_engine src/build_engine_main.cpp)
target_link_libraries(r1_build_engine trt_policy)

add_executable(r1_parity_check src/parity_check_main.cpp)
target_link_libraries(r1_parity_check trt_policy)

add_executable(r1_policy_node src/policy_node.cpp)
target_link_libraries(r1_policy_node trt_policy)
ament_target_dependencies(r1_policy_node rclcpp std_msgs)

install(TARGETS trt_policy
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS r1_build_engine r1_parity_check r1_policy_node
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME} OPTIONAL)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_obs_assembler test/test_obs_assembler.cpp)
  if(TARGET test_obs_assembler)
    target_include_directories(test_obs_assembler PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  endif()
endif()

ament_package()
