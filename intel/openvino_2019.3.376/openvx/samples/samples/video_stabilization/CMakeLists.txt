#include definition for common options
include(options)

if (BUILDING_SAMPLE_NAME AND NOT "${BUILDING_SAMPLE_NAME}" STREQUAL "OFF" AND NOT "${BUILDING_SAMPLE_NAME}" STREQUAL "video_stabilization")
    message(STATUS "video_stabilization skipped")
    return()
endif()

#request OpenVX libs
find_package(OpenVX)

#try to use openCV if available
find_package(SampleOpenCV)
if(NOT OpenCV_FOUND)
    message(WARNING "video_stabilization sample will not be built due to OpenCV absence")
    return()
endif()

#try to use ITT if available
find_package(INTEL_ITT)


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
# The module and the final binary use video_stabilization_core library that is also built
# separately

set(USER_NODES video_stabilization_user_nodes)

# Video stabilization core library
set(CORE_LIB video_stabilization_core)
add_library(${CORE_LIB} ${CORE_LIB}.cpp debug_visualization_lib.cpp)
target_link_libraries(${CORE_LIB} common ${OpenCV_LIBS} ${OpenVX_LIBS} ${INTEL_ITT_LIBS})
# use -fPIC because this library will be used as a part of another library
set_target_properties(${CORE_LIB} PROPERTIES COMPILE_FLAGS "-fPIC -pthread")
set_target_properties(${CORE_LIB} PROPERTIES LINK_FLAGS "-pthread")

# VX and VXU functions for user-defined nodes to make graph construction easier
set(USER_NODES_LIB ${USER_NODES}_lib)
add_library(${USER_NODES_LIB} ${USER_NODES_LIB}.cpp)
# use -fPIC because this library will be used as a part of another library
set_target_properties(${USER_NODES_LIB} PROPERTIES COMPILE_FLAGS "-fPIC -pthread")
set_target_properties(${USER_NODES_LIB} PROPERTIES LINK_FLAGS "-pthread")

# Dynamically loadable module with video stabilization user node kernels implementation
# This module should be explicitly loaded by vxLoadKernels.
set(USER_NODES_MODULE ${USER_NODES}_module)
add_library(${USER_NODES_MODULE} SHARED ${USER_NODES_MODULE}.cpp)
target_link_libraries (${USER_NODES_MODULE} common ${OpenVX_LIBS} ${INTEL_ITT_LIBS} ${CORE_LIB})
set_target_properties(${USER_NODES_MODULE} PROPERTIES COMPILE_FLAGS "-pthread")
set_target_properties(${USER_NODES_MODULE} PROPERTIES LINK_FLAGS "-pthread")


# Sample application executable
set(BINARY video_stabilization)
set(SOURCES video_stabilization.cpp video_stabilization_opencv.cpp cmdoptions.cpp video_stabilization_openvx.cpp)

add_executable(${BINARY} ${SOURCES} ${COMMON_SOURCES})
target_link_libraries (${BINARY} common ${USER_NODES_LIB} ${OpenCV_LIBS} ${OpenVX_LIBS} ${INTEL_ITT_LIBS} ${CORE_LIB})
set_target_properties(${BINARY} PROPERTIES COMPILE_FLAGS "-pthread")
set_target_properties(${BINARY} PROPERTIES LINK_FLAGS "-pthread")
add_dependencies(${BINARY} ${USER_NODES_MODULE})

# Install sample binaries
install(TARGETS ${USER_NODES_MODULE} ${BINARY} ${CORE_LIB}
    DESTINATION bin/video_stabilization)

install (FILES toy_flower.mp4 DESTINATION "bin/video_stabilization")

set(HETERO_CONFIGS
    hetero.config.cpu-harris-pyramid.txt
    hetero.config.cpu-harris.txt
    hetero.config.cpu-optical-flow.txt
    hetero.config.cpu-pyramid.txt
    hetero.config.gpu-all.txt
    hetero.config.cpu-all.txt
    hetero.config.gpu-first-part.txt
    hetero.config.gpu-last-part.txt
    hetero.config.cpu-convert-extract.txt
    hetero.config.gpu-harris-gaussian-optical.txt
    hetero.config.gpu-harris-2-warps.txt
    hetero.config.gpu-harris-3-warps.txt
    hetero.config.gpu-harris-warp.txt 
)

install (FILES ${HETERO_CONFIGS} DESTINATION "bin/video_stabilization")

