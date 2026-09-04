cmake_minimum_required(VERSION 3.10)
project(r1_probe_lowstate CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(probe_lowstate probe_lowstate.cpp)

# The SDK normally installs a package config. Use it when it is there; fall
# back to the plain install layout when it is not, so a missing config file
# does not block the one measurement W06 is waiting on.
find_package(unitree_sdk2 QUIET)
if(unitree_sdk2_FOUND)
  message(STATUS "using unitree_sdk2 package config")
  target_link_libraries(probe_lowstate unitree_sdk2)
else()
  message(WARNING "unitree_sdk2 package config not found; linking by path")
  target_include_directories(probe_lowstate PRIVATE /usr/local/include)
  find_library(UNITREE_SDK2 unitree_sdk2 PATHS /usr/local/lib REQUIRED)
  find_library(DDSCXX ddscxx PATHS /usr/local/lib REQUIRED)
  find_library(DDSC   ddsc   PATHS /usr/local/lib REQUIRED)
  target_link_libraries(probe_lowstate ${UNITREE_SDK2} ${DDSCXX} ${DDSC})
endif()

find_package(Threads REQUIRED)
target_link_libraries(probe_lowstate Threads::Threads)
