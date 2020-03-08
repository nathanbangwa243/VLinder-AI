get_filename_component(SAMPLE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Check if project is not defined and this CMakeFiles.txt is root.
# In this case
# 1. define the project,
# 2. add 'cmake' into CMAKE_MODULE_PATH
# 3. call necessary cmake_minimum_required()

set(NAME auto_contrast)
set(BINARY auto_contrast)
set(SOURCES auto_contrast.cpp)

if ( BUILDING_SAMPLE_NAME AND NOT "${BUILDING_SAMPLE_NAME}" STREQUAL "OFF" AND NOT "${BUILDING_SAMPLE_NAME}" STREQUAL "${NAME}")
    message(STATUS "${NAME} skipped")
    return()
endif()

if(NOT DEFINED PROJECT_NAME)
    message("${SAMPLE_NAME} alone")
    cmake_minimum_required(VERSION 2.8.11)
    project("${SAMPLE_NAME}_project")
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../../cmake")
endif()

#include definition for common options
include(options)
#request OpenVX libs
find_package(OpenVX)
#try using OpenCV if available
find_package(SampleOpenCV)
#request ITT libs because common lib may request it
find_package(INTEL_ITT)

include_directories(${OpenVX_INCLUDE})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${INTEL_ITT_INCLUDE_DIRS})

set(common_INCLUDE_DIRS "../common/include")
include_directories(${common_INCLUDE_DIRS})

# Define common sources to use performance profiler and other staff
set(COMMON_SOURCES
    ../common/src/basic.cpp
    ../common/src/helper.cpp
    ../common/src/perfprof.cpp
    ../common/src/cmdparser.cpp
)

add_definitions(-DINTEL_SAMPLE_PERFPROF_STDOUT_DEBUG)

add_executable(${BINARY} ${SOURCES} ${COMMON_SOURCES} )
set_target_properties(${BINARY} PROPERTIES COMPILE_FLAGS "-pthread")
set_target_properties(${BINARY} PROPERTIES LINK_FLAGS "-pthread")

#link with opencv and openvx
target_link_libraries (${BINARY} ${OpenCV_LIBS} ${OpenVX_LIBS} ${INTEL_ITT_LIBS})


#Install sample binaries
install(TARGETS ${BINARY}
     DESTINATION "bin/${NAME}")
install (FILES low_contrast_vga.jpg DESTINATION "bin/${NAME}")

