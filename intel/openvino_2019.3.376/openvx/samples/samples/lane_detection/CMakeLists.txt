#check if cmake was run for no sample root folder
if(NOT DEFINED PROJECT_NAME)
    message(STATUS "lane detection alone")
    cmake_minimum_required(VERSION 2.8.11)
    project(lane_detection_project)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake")
endif()

if (BUILDING_SAMPLE_NAME AND NOT "${BUILDING_SAMPLE_NAME}" STREQUAL "OFF" AND NOT "${BUILDING_SAMPLE_NAME}" STREQUAL "lane_detection")
    message(STATUS "lane_detection skipped")
    return()
endif()



#Include common options
include(options)

#OpenVX libs
find_package(OpenVX)
include_directories(${OpenVX_INCLUDE})
set(LIBS ${LIBS} ${OpenVX_LIBS})

#OpenCV libs are needed to implement reference pipline
find_package(SampleOpenCV)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${OpenCV_LIBS})
else()
    message(WARNING "lane_detection sample will not be built due to OpenCV absence")
    return()
endif()

find_package(INTEL_ITT)
include_directories(${INTEL_ITT_INCLUDE_DIRS})
#set(LIBS ${LIBS} ${INTEL_ITT_LIBS})

# Define common sources to use performance profiler and other staff
include_directories(../common/include)
set(COMMON_SOURCES
    ../common/src/basic.cpp
    ../common/src/helper.cpp
    ../common/src/perfprof.cpp
    ../common/src/cmdparser.cpp
)

#add common sample lib to use performance profiler
add_executable(lane_detection
	lane_detection.cpp
	collect_lane_marks.cpp
	${COMMON_SOURCES})
include_directories(../common/include)

set_target_properties(lane_detection PROPERTIES COMPILE_FLAGS "-pthread")
set_target_properties(lane_detection PROPERTIES LINK_FLAGS "-pthread")

#link with opencv and openvx and other libs
target_link_libraries(lane_detection ${LIBS} ${INTEL_ITT_LIBS})

# Install sample binary
install(TARGETS lane_detection DESTINATION bin/lane_detection)
install(FILES road_lane.mp4 DESTINATION bin/lane_detection)

#create dynamic library with user node implements final postprocessing and drawing
add_library(lane_detection_nodes SHARED
    lane_detection_user_nodes_module.cpp
    collect_lane_marks.cpp
    ${COMMON_SOURCES})
target_link_libraries(lane_detection_nodes ${LIBS} ${INTEL_ITT_LIBS})
set_target_properties(lane_detection_nodes  PROPERTIES LINK_FLAGS "-Wl,-Bsymbolic -Wl,--version-script=\"${CMAKE_CURRENT_SOURCE_DIR}/lane_detection_nodes.map\"")
install(TARGETS lane_detection_nodes DESTINATION bin/lane_detection)
install(FILES lane_detection.graphml DESTINATION bin/lane_detection)

