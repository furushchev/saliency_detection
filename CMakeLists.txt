cmake_minimum_required(VERSION 2.8.3)
project(saliency_detection)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_transport
#  opencv2
  roscpp
  sensor_msgs
)

find_package(OpenCV REQUIRED)
catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES saliency_detection
  CATKIN_DEPENDS cv_bridge geometry_msgs image_transport opencv2 roscpp sensor_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Your package locations should be listed before other locations
include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(saliencyDetectionHou src/saliencyDetectionHou.cpp)
target_link_libraries(saliencyDetectionHou
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS})
add_executable(saliencyDetectionRudinac src/saliencyDetectionRudinac.cpp)
target_link_libraries(saliencyDetectionRudinac
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS})
add_executable(saliencyDetectionItti src/saliencyDetectionItti.cpp src/cvgabor.cpp)
target_link_libraries(saliencyDetectionItti
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS})


#############
## Install ##
#############

## Mark executables and/or libraries for installation
install(TARGETS saliencyDetectionHou saliencyDetectionRudinac saliencyDetectionItti
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  PATTERN ".svn" EXCLUDE
)

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_saliency_detection.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
