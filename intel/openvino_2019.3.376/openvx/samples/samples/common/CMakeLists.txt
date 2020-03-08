include(options)
find_package(OpenVX)
find_package(SampleOpenCV)
find_package(INTEL_ITT)
add_library (common src/perfprof.cpp src/cmdparser.cpp src/basic.cpp src/helper.cpp src/hetero.cpp)

set_target_properties(common PROPERTIES COMPILE_FLAGS "-fPIC")

include_directories(${OpenVX_INCLUDE})
include_directories(${INTEL_ITT_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories (common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)


