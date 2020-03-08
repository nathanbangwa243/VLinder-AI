if (BUILDING_SAMPLE_NAME AND NOT "${BUILDING_SAMPLE_NAME}" STREQUAL "OFF" AND NOT "${BUILDING_SAMPLE_NAME}" STREQUAL "ocl_custom_kernel")
    message(STATUS "ocl_custom_kernel skipped")
    return()
endif()

#check if cmake was run for no sample root folder
if(NOT DEFINED PROJECT_NAME)
    message(STATUS "ocl_custom_kernel alone")
    cmake_minimum_required(VERSION 2.8.11)
    project(ocl_custom_kernel_project)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake")
endif()

#include common options
include(options)

#OpenVX libs
find_package(OpenVX)
include_directories(${OpenVX_INCLUDE})
set(LIBS ${LIBS} ${OpenVX_LIBS})

#OpenCV libs are needed to read input video and output results of processing
find_package(SampleOpenCV)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${OpenCV_LIBS})
else()
    message(WARNING "ocl_custom_kernel sample will not be built due to OpenCV absence")
    return()
endif()

set(COMMON_SOURCES
    ../common/src/basic.cpp
    ../common/src/helper.cpp
    ../common/src/perfprof.cpp
    ../common/src/cmdparser.cpp
)

#add common sample lib to use performance profiler
add_executable(ocl_custom_kernel ocl_custom_kernel.cpp ${COMMON_SOURCES} )
include_directories(../common/include)
set(LIBS ${LIBS} common)

find_package(INTEL_ITT)
include_directories(${INTEL_ITT_INCLUDE_DIRS})
set(LIBS ${LIBS} ${INTEL_ITT_LIBS})

set_target_properties(ocl_custom_kernel PROPERTIES COMPILE_FLAGS "-pthread")
set_target_properties(ocl_custom_kernel PROPERTIES LINK_FLAGS "-pthread")

#link with opencv and openvx and other libs
target_link_libraries(ocl_custom_kernel ${LIBS} )

# Install sample binary
install(TARGETS ocl_custom_kernel DESTINATION bin/ocl_custom_kernel)
install(FILES toy_flower.mp4 DESTINATION bin/ocl_custom_kernel)
install(FILES source.cl DESTINATION bin/ocl_custom_kernel)

