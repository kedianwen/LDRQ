cmake_minimum_required(VERSION 3.10)
project(r1_hw_bridge)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Why this file is shaped like this
#
# Unitree installs CycloneDDS 0.10.2 headers under /usr/local/include/dds/ and
# /usr/local/include/ddscxx/. ROS foxy installs CycloneDDS 0.7.0 under
# /opt/ros/foxy/include/dds/. A target that sees both prefixes resolves each
# header from whichever comes first in the search order, so the compile picks up
# a blend of the two versions -- observed as `ddsi_sertype does not name a type`
# and a dozen similar errors from inside the SDK's own headers.
#
# Ordering tricks make that go away by luck. The split below makes it
# impossible: robot_io.cpp is the only file that touches the SDK and is built
# with no ROS on its include path, the node is built with no /usr/local/include
# on its include path, and the two meet through the POD types in robot_io.hpp.
# The property filters at the bottom enforce it even if a vendor CMake config
# injects a directory-scope include_directories() behind our back.
# ---------------------------------------------------------------------------

set(R1_SDK_PREFIX "/usr/local" CACHE PATH "unitree_sdk2 install prefix")

# --- the SDK side: no ROS -----------------------------------------------
add_library(r1_robot_io STATIC src/robot_io.cpp)
target_include_directories(r1_robot_io PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${R1_SDK_PREFIX}/include
  ${R1_SDK_PREFIX}/include/ddscxx)

# NO_DEFAULT_PATH is load-bearing. `PATHS` is a FALLBACK: CMake searches the
# default prefixes (which include /opt/ros/<distro> via CMAKE_PREFIX_PATH)
# first, so a plain find_library(ddsc PATHS /usr/local/lib) resolves to ROS's
# CycloneDDS 0.7.0. Linking the SDK's libddscxx.so (0.10.2) against it fails
# with a wall of `undefined reference to dds_qset_data_representation` and
# friends -- the link-time twin of the header clash handled above.
find_library(UNITREE_SDK2 unitree_sdk2 PATHS ${R1_SDK_PREFIX}/lib NO_DEFAULT_PATH REQUIRED)
find_library(DDSCXX ddscxx PATHS ${R1_SDK_PREFIX}/lib NO_DEFAULT_PATH REQUIRED)
find_library(DDSC   ddsc   PATHS ${R1_SDK_PREFIX}/lib NO_DEFAULT_PATH REQUIRED)
message(STATUS "r1_hw_bridge DDS: ${DDSCXX} / ${DDSC}")

# Order matters: ddscxx depends on ddsc.
# PRIVATE on a static library propagates the libraries for linking but NOT the
# include directories -- which is exactly the boundary we need.
target_link_libraries(r1_robot_io PRIVATE
  ${UNITREE_SDK2} ${DDSCXX} ${DDSC} Threads::Threads)

# --- the ROS side: no SDK -----------------------------------------------
add_executable(r1_hw_bridge_node src/bridge_node.cpp)
target_include_directories(r1_hw_bridge_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
ament_target_dependencies(r1_hw_bridge_node rclcpp std_msgs)
target_link_libraries(r1_hw_bridge_node r1_robot_io Threads::Threads)

# The same version choice has to hold at load time. `--disable-new-dtags` emits
# DT_RPATH rather than DT_RUNPATH, and DT_RPATH is searched BEFORE
# LD_LIBRARY_PATH -- so sourcing a ROS setup.bash later cannot swap CycloneDDS
# 0.10.2 for 0.7.0 underneath a running bridge. env.sh also orders the path,
# but that is a convention; this is a guarantee.
set_property(TARGET r1_hw_bridge_node APPEND_STRING
  PROPERTY LINK_FLAGS " -Wl,--disable-new-dtags")
set_target_properties(r1_hw_bridge_node PROPERTIES
  INSTALL_RPATH "${R1_SDK_PREFIX}/lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE)

# --- enforce the split ---------------------------------------------------
function(r1_forbid_includes target regex what)
  get_target_property(_inc ${target} INCLUDE_DIRECTORIES)
  if(_inc)
    set(_kept ${_inc})
    list(FILTER _kept EXCLUDE REGEX "${regex}")
    if(NOT "${_kept}" STREQUAL "${_inc}")
      message(STATUS "${target}: dropped ${what} include paths (DDS version split)")
      set_target_properties(${target} PROPERTIES INCLUDE_DIRECTORIES "${_kept}")
    endif()
  endif()
endfunction()

r1_forbid_includes(r1_hw_bridge_node "^${R1_SDK_PREFIX}/include" "SDK")
r1_forbid_includes(r1_robot_io "^/opt/ros" "ROS")

install(TARGETS r1_hw_bridge_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  # Same reasoning as r1_policy_runner: a ros-base image without gtest must not
  # fail the whole build, and "0 tests" must not read as "0 failures".
  find_package(ament_cmake_gtest QUIET)
  if(ament_cmake_gtest_FOUND)
    ament_add_gtest(test_joint_map test/test_joint_map.cpp)
    target_include_directories(test_joint_map PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  else()
    message(WARNING "ament_cmake_gtest not found -- joint map tests SKIPPED")
  endif()
endif()

ament_package()
