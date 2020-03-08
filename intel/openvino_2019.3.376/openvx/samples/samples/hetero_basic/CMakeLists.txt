find_package(OpenVX)

if (BUILDING_SAMPLE_NAME AND NOT "${BUILDING_SAMPLE_NAME}" STREQUAL "OFF" AND NOT "${BUILDING_SAMPLE_NAME}" STREQUAL "hetero_basic")
    message(STATUS "hetero_basic skipped")
	return()
endif()

#include common options
include(options)

#OpenCV libs are needed to read input video and output results of processing
find_package(SampleOpenCV)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${OpenCV_LIBS})
else()
    message(WARNING "hetero_basic sample will not be built due to OpenCV absence")
    return()
endif()

# Define common sources to use performance profiler and other staff
set(COMMON_SOURCES
    ../common/src/basic.cpp
    ../common/src/helper.cpp
    ../common/src/perfprof.cpp
    ../common/src/cmdparser.cpp
)

#add common sample lib to use performance profiler
add_executable(hetero_basic hetero_basic.cpp ${COMMON_SOURCES} )
include_directories(../common/include)
set(LIBS ${LIBS} common)

#OpenVX libs
find_package(OpenVX)
include_directories(${OpenVX_INCLUDE})
set(LIBS ${LIBS} ${OpenVX_LIBS})

find_package(INTEL_ITT)
include_directories(${INTEL_ITT_INCLUDE_DIRS})
set(LIBS ${LIBS} ${INTEL_ITT_LIBS})

set_target_properties(hetero_basic PROPERTIES COMPILE_FLAGS "-pthread")
set_target_properties(hetero_basic PROPERTIES LINK_FLAGS "-pthread")

#link with opencv and openvx and other libs
target_link_libraries(hetero_basic ${LIBS} )

# Install sample binary
install(TARGETS hetero_basic DESTINATION bin/hetero_basic)
install(FILES toy_flower_512x512.mp4 DESTINATION bin/hetero_basic)

