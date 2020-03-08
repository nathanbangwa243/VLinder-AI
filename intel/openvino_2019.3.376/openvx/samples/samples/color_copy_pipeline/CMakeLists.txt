#include common options
include(options)

set(NAME color_copy_pipeline)
set(NAME_LIB color_copy_pipeline_lib)
set(NAME_IPA_LIB ipa)

if (BUILDING_SAMPLE_NAME AND NOT "${BUILDING_SAMPLE_NAME}" STREQUAL "OFF" AND NOT "${BUILDING_SAMPLE_NAME}" STREQUAL "${NAME}")
    message(STATUS "${NAME} skipped")
    return()
endif()

add_executable(${NAME} main.cpp pipelinecontrol.cpp
	../common/src/basic.cpp
	../common/src/helper.cpp
	../common/src/perfprof.cpp
	../common/src/cmdparser.cpp
    vxbackgroundsuppress_lib.cpp
    vxneutralpixeldetection_lib.cpp
    vxremovefringe_lib.cpp
    vxsymm7x7_opencl_lib.cpp
    vxremovefringe_opencl_lib.cpp
    vxipahalftone_lib.cpp
    vxgainoffset10_lib.cpp
    vxgainoffset12_lib.cpp
    vxgenedgemask_lib.cpp
    vxgenedgek_lib.cpp
	)

#custom kernels library
add_library(${NAME_LIB} SHARED
        vxbackgroundsuppress.cpp
        vxneutralpixeldetection.cpp
        vxremovefringe.cpp
        vxsymm7x7_opencl_module.cpp
        vxremovefringe_opencl_module.cpp
        vxipahalftone.cpp
        vxgainoffset10.cpp
        vxgainoffset12.cpp
        vxgenedgemask.cpp
        vxgenedgek.cpp
        vxpublishkernels.cpp
        ../common/src/helper.cpp
)

include_directories(ipa)
if (UNIX)
add_library( ${NAME_IPA_LIB} SHARED
             ipa/doubler.c
             ipa/gradient.c
             ipa/halftone.c
             ipa/rescale.c
             ipa/rotate.c
             ipa/transparency.c
             ipa/ipa.c
           )
else()
add_library( ${NAME_IPA_LIB}
             ipa/doubler.c
             ipa/gradient.c
             ipa/halftone.c
             ipa/rescale.c
             ipa/rotate.c
             ipa/transparency.c
             ipa/ipa.c
           )
endif()

set_target_properties(ipa PROPERTIES COMPILE_FLAGS "-msse4.2 ")

#add common sample lib to use performance profiler
include_directories(../common/include)

find_package(INTEL_ITT)
include_directories(${INTEL_ITT_INCLUDE_DIRS})

#add common sample lib to use performance profiler
include_directories(../common/include)
set(LIBS ${LIBS} common ${INTEL_ITT_LIBS})

#OpenVX libs
find_package(OpenVX)
include_directories(${OpenVX_INCLUDE})
set(LIBS ${LIBS} ${OpenVX_LIBS})

set_target_properties(${NAME} PROPERTIES COMPILE_FLAGS "-pthread")
set_target_properties(${NAME} PROPERTIES LINK_FLAGS "-pthread")
#custom kernels are optimized using SSE intrinsics
set_target_properties(${NAME_LIB} PROPERTIES COMPILE_FLAGS "-msse4.1")

#link with openvx and other libs
target_link_libraries(${NAME} ${LIBS} )
target_link_libraries(${NAME_LIB} ${LIBS})
target_link_libraries(${NAME_IPA_LIB} ${LIBS})
target_link_libraries(${NAME_LIB} ${NAME_IPA_LIB})

# Install OpenCL source code
install(FILES vxsymm7x7_opencl_impl.cl DESTINATION "bin/${NAME}")
install(FILES vxremovefringe_opencl_impl.cl DESTINATION "bin/${NAME}")

# Install sample binary
install(TARGETS ${NAME} DESTINATION bin/${NAME})
install(TARGETS ${NAME_IPA_LIB} DESTINATION bin/${NAME})
install(TARGETS ${NAME_LIB} DESTINATION bin/${NAME})
install (FILES low_contrast_5120x6592_I444.raw DESTINATION bin/${NAME})

