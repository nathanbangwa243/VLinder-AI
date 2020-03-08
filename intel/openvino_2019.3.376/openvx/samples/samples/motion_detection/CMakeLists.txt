#include definition for common options
include(options)

if ( BUILDING_SAMPLE_NAME AND NOT "${BUILDING_SAMPLE_NAME}" STREQUAL "OFF" AND NOT "${BUILDING_SAMPLE_NAME}" STREQUAL "motion_detection")
    message(STATUS "motion_detection skipped")
    return()
endif()

#try to use openCV if available
find_package(SampleOpenCV)
if(NOT OpenCV_FOUND)
    message(WARNING "motion_detection sample will not be built due to OpenCV absence")
    return()
endif()

#try to use ITT if available
find_package(INTEL_ITT)
#request OpenVx libs
find_package(OpenVX)

include_directories(${OpenVX_INCLUDE})
include_directories(${IAP_INTERNAL_INCLUDES})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${INTEL_ITT_INCLUDE_DIRS})

set(common_INCLUDE_DIRS "../common/include")
include_directories(${common_INCLUDE_DIRS})

# Define common sources to use performance profiler and other staff
set(COMMON_SOURCES
    ../common/src/basic.cpp
    ../common/src/helper.cpp
    ../common/src/hetero.cpp
    ../common/src/perfprof.cpp
    ../common/src/cmdparser.cpp
)

# Code with user nodes is organized in two pieces: lib and module
#  - lib is linked directly to the final binary executable
#  - module is built as shared library and loaded by executable on demand with vxLoadKernels
# The module and the final binary use motion_detection_core library that is also built
# separately

set(USER_NODES motion_detection_user_nodes)

# Motion detection core library
set(CORE_LIB motion_detection_core)
add_library(${CORE_LIB} ${CORE_LIB}.cpp)
target_link_libraries(${CORE_LIB} ${OpenCV_LIBS} ${OpenVX_LIBS} ${INTEL_ITT_LIBS})
# use -fPIC because this library will be used as a part of another library
set_target_properties(${CORE_LIB} PROPERTIES COMPILE_FLAGS "-fPIC")

# VX and VXU functions for user-defined nodes to make graph construction easier
set(USER_NODES_LIB ${USER_NODES}_lib)
add_library(${USER_NODES_LIB} ${USER_NODES_LIB}.cpp)
target_link_libraries (${USER_NODES_LIB} ${OpenCV_LIBS} ${OpenVX_LIBS} ${INTEL_ITT_LIBS} )
# use -fPIC because this library will be used as a part of another library
set_target_properties(${USER_NODES_LIB} PROPERTIES COMPILE_FLAGS "-fPIC")

# Dynamically loadable module with motion detection user node kernels implementation
# This module should be explicitly loaded by vxLoadKernels.
set(USER_NODES_MODULE ${USER_NODES}_module)
add_library(${USER_NODES_MODULE} SHARED ${USER_NODES_MODULE}.cpp)
set_target_properties(${USER_NODES_MODULE} PROPERTIES COMPILE_FLAGS "-fPIC")
target_link_libraries (${USER_NODES_MODULE} common ${OpenCV_LIBS} ${OpenVX_LIBS} ${INTEL_ITT_LIBS} ${CORE_LIB})

# Sample application executable
set(BINARY motion_detection)
set(SOURCES motion_detection.cpp cmdoptions.cpp motion_detection_common.cpp)

add_executable(${BINARY} ${SOURCES} ${COMMON_SOURCES})
set_target_properties(${BINARY} PROPERTIES COMPILE_FLAGS "-pthread")
set_target_properties(${BINARY} PROPERTIES LINK_FLAGS "-pthread")
target_link_libraries (${BINARY} ${USER_NODES_LIB} ${OpenCV_LIBS} ${OpenVX_LIBS} ${CORE_LIB})
add_dependencies(${BINARY} ${USER_NODES_MODULE})

# Dynamically loadable shared object with motion detection functionality
# This shared object will be loaded by gstreamer plugin
set(WORKLOAD_PLUGIN_MODULE wl_md)
if(NOT WIN32)
add_library(${WORKLOAD_PLUGIN_MODULE} SHARED libwl_md.cpp motion_detection_common.cpp)
else(WIN32)
add_library(${WORKLOAD_PLUGIN_MODULE} SHARED motion_detection_common.cpp)
endif()
target_link_libraries (${WORKLOAD_PLUGIN_MODULE} ${USER_NODES_LIB} ${USER_NODES_MODULE} ${OpenCV_LIBS} ${OpenVX_LIBS} ${CORE_LIB})

# Install sample binaries
install(TARGETS ${USER_NODES_MODULE} ${BINARY} ${WORKLOAD_PLUGIN_MODULE}
    DESTINATION bin/motion_detection)

install (FILES hetero.config.cpu-all.txt DESTINATION bin/motion_detection)
install (FILES hetero.config.default.txt DESTINATION bin/motion_detection)

install (FILES VID_640x360.mp4  DESTINATION "bin/motion_detection")

