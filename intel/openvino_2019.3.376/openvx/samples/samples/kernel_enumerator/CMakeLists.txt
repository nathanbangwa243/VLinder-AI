#include common options
include(options)

if (BUILDING_SAMPLE_NAME AND NOT "${BUILDING_SAMPLE_NAME}" STREQUAL "OFF" AND NOT "${BUILDING_SAMPLE_NAME}" STREQUAL "kernel_enumerator" )
    message(STATUS "kernel_enumerator skipped")
    return()
endif()

#OpenVX libs
find_package(OpenVX)
include_directories(${OpenVX_INCLUDE})
set(LIBS ${LIBS} ${OpenVX_LIBS})


add_executable(kernel_enumerator kernel_enumerator.cpp ../common/src/helper.cpp ../common/src/basic.cpp ../common/src/perfprof.cpp)

#add common sample lib to use performance profiler
include_directories(../common/include)

find_package(INTEL_ITT)
include_directories(${INTEL_ITT_INCLUDE_DIRS})
set(LIBS ${LIBS} ${INTEL_ITT_LIBS})

set_target_properties(kernel_enumerator PROPERTIES COMPILE_FLAGS "-pthread")
set_target_properties(kernel_enumerator PROPERTIES LINK_FLAGS "-pthread")

target_link_libraries(kernel_enumerator ${LIBS} )

# Install sample binary
install(TARGETS kernel_enumerator DESTINATION bin/kernel_enumerator)

