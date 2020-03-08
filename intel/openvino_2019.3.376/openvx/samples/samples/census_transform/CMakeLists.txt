#include common options
include(options)

set(NAME census_transform)
set(NAME_LIB census_transform_lib)

if (BUILDING_SAMPLE_NAME AND NOT "${BUILDING_SAMPLE_NAME}" STREQUAL "OFF" AND NOT "${BUILDING_SAMPLE_NAME}" STREQUAL "${NAME}")
    message(STATUS "${NAME} skipped")
    return()
endif()

#OpenVX libs
find_package(OpenVX)
include_directories(${OpenVX_INCLUDE})
set(LIBS ${LIBS} ${OpenVX_LIBS})

#OpenCV libs are needed to implement reference pipline
find_package(SampleOpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBS})
if( NOT OpenCV_FOUND )
    message(WARNING "${NAME} is not built because OpenCV is not detected")
    return()
endif( )

set(HOST_SOURCES
    main.cpp
    vx_censustransform_lib.c
    vx_censustransformtiled_lib.c
    vx_censustransform_opencl_lib.cpp
)

set(LIB_SOURCES
    vx_censustransform_impl.c
    vx_censustransform_module.c
    vx_censustransformtiled_module.c
    vx_censustransform_opencl_module.cpp
)

# Define common sources to use performance profiler and other staff
set(COMMON_SOURCES
    ../common/src/basic.cpp
    ../common/src/helper.cpp
    ../common/src/perfprof.cpp
    ../common/src/cmdparser.cpp
)

add_executable(${NAME} ${HOST_SOURCES} ${COMMON_SOURCES} )
add_library(${NAME_LIB} SHARED ${LIB_SOURCES})

find_package(INTEL_ITT)
include_directories(${INTEL_ITT_INCLUDE_DIRS})

#add common sample lib to use performance profiler
include_directories(../common/include)
set(LIBS ${LIBS} common)



set_target_properties(${NAME} PROPERTIES COMPILE_FLAGS "-pthread")
set_target_properties(${NAME} PROPERTIES LINK_FLAGS "-pthread")

#link with opencv and openvx and other libs
target_link_libraries(${NAME} ${LIBS} ${INTEL_ITT_LIBS})
target_link_libraries(${NAME_LIB} ${LIBS} ${INTEL_ITT_LIBS})

install (FILES toy_flower.mp4 DESTINATION "bin/${NAME}")

# Install OpenCL source code
install(FILES vx_censustransform_opencl_impl.cl DESTINATION "bin/${NAME}")

# Install sample binary
install(TARGETS ${NAME} DESTINATION bin/${NAME})
install(TARGETS ${NAME_LIB} DESTINATION bin/${NAME})

