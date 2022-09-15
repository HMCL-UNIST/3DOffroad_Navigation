cmake_minimum_required(VERSION 3.8)
project(autorally_control)

find_package(catkin REQUIRED COMPONENTS
  autorally_msgs
  cmake_modules
  dynamic_reconfigure
  geometry_msgs  
  nav_msgs  
  roscpp
  sensor_msgs
  std_msgs
  tf
)

find_package(Boost 1.55.0 REQUIRED)
find_package(Eigen3 REQUIRED)

set(BUILD_FLAGS "-std=c++11 -Wuninitialized")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BUILD_FLAGS}")

generate_dynamic_reconfigure_options(cfg/gpsWaypoint_params.cfg
				     cfg/PathIntegralParams.cfg)

catkin_package(
  DEPENDS Boost
  CATKIN_DEPENDS roscpp std_msgs geometry_msgs nav_msgs dynamic_reconfigure autorally_msgs
  INCLUDE_DIRS include
  LIBRARIES param_getter
)

include_directories(include)
include_directories(SYSTEM ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${Boost_INCLUDE_DIR})


add_subdirectory(src/joystick)
add_subdirectory(src/gpsWaypoint)


install(DIRECTORY launch/
   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
   FILES_MATCHING PATTERN "*.launch" PATTERN "*.machine" PATTERN "*.yaml"
)

install(DIRECTORY include/${PROJECT_NAME}/
   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY cfg/cpp/${PROJECT_NAME}/
   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
   FILES_MATCHING PATTERN "*.h"
)


