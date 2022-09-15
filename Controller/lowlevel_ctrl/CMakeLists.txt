cmake_minimum_required(VERSION 3.0.2)
project(lowlevel_ctrl)

find_package(Eigen3 REQUIRED)

if(NOT EIGEN3_FOUND)
  # Fallback to cmake_modules
  find_package(cmake_modules REQUIRED)
  find_package(Eigen REQUIRED)
  set(EIGEN3_INCLUDE_DIRS ${EIGEN_INCLUDE_DIRS})
  set(EIGEN3_LIBRARIES ${EIGEN_LIBRARIES})  # Not strictly necessary as Eigen is head only
  # Possibly map additional variables to the EIGEN3_ prefix.
else()
  set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
endif()

find_package(catkin REQUIRED COMPONENTS  
  nav_msgs  
  roscpp
  sensor_msgs
  geometry_msgs
  visualization_msgs  
  std_msgs
  tf  
  dynamic_reconfigure  
  autorally_msgs
  hmcl_msgs
)

generate_dynamic_reconfigure_options(
     cfg/test.cfg
)

catkin_package(
  CATKIN_DEPENDS nav_msgs roscpp sensor_msgs geometry_msgs std_msgs hmcl_msgs tf visualization_msgs dynamic_reconfigure autorally_msgs
  LIBRARIES
)

include_directories(
  include  
  ${dynamic_reconfigure_PACKAGE_PATH}/cmake/cfgbuild.cmake
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
add_executable(dynamic_reconfigure_node src/utils.cpp src/lowpass_filter.cpp src/lowlevel_ctrl.cpp src/trajectory.cpp)
add_dependencies(dynamic_reconfigure_node ${PROJECT_NAME}_gencfg)
target_link_libraries(dynamic_reconfigure_node ${catkin_LIBRARIES})
SET(CMAKE_CXX_FLAGS "-O2 -g -Wall ${CMAKE_CXX_FLAGS}")

add_executable(lowlevel_ctrl src/utils.cpp src/lowpass_filter.cpp src src/lowlevel_ctrl.cpp src/trajectory.cpp)
add_dependencies(lowlevel_ctrl ${catkin_EXPORTED_TARGETS})
target_link_libraries(lowlevel_ctrl ${catkin_LIBRARIES})



install(
  TARGETS
  lowlevel_ctrl    
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)