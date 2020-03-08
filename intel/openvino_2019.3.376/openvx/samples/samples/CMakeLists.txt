#check if cmake was run for no sample root folder
if(NOT DEFINED PROJECT_NAME)
    message(STATUS "INTEL_VX_SAMPLES alone")
    cmake_minimum_required(VERSION 2.8.11)
    project(INTEL_VX_SAMPLES)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../cmake")
endif()

#set "Release" build for all samples build
if ( NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
    set(CMAKE_BUILD_TYPE "Release")
    message( STATUS "\nChanged from default to Release build type" )
else()
    message( STATUS "\nBuild type is ${CMAKE_BUILD_TYPE}" )
endif()

find_package(OpenVX)
message( STATUS "\ncommon library" )
add_subdirectory(common)


# find existing configuration file if the specific one is not defined
if (NOT DEFINED SAMPLES_TO_PACK_CONFIG)
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cv_samples.config)
        set(SAMPLES_TO_PACK_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/cv_samples.config)
    else()
        file(GLOB configs_available ${CMAKE_CURRENT_SOURCE_DIR}/ *.config)
        list(GET configs_available 0 SAMPLES_TO_PACK_CONFIG)
    endif()
endif()

message(STATUS "Using file '${SAMPLES_TO_PACK_CONFIG}' with the list of samples")
# read list of samples to build/pack
file(STRINGS ${SAMPLES_TO_PACK_CONFIG} samples_to_pack)
foreach(sample ${samples_to_pack})
    list(GET sample 0 sample_name)
    list(LENGTH sample sample_list_length)
    if (NOT (${sample_list_length} EQUAL 1))
        list(GET sample 1 sample_condition)
    endif()
    if (NOT sample_condition)
        set(sample_condition "1 EQUAL 1")
    endif()
    message(STATUS "Sample '${sample_name}' condition is '${sample_condition}'")
    eval(${sample_condition})
    #cut
    if (BUILD_FOR_PACKING)
        set(eval_result ON)
    endif()
    #uncut
    if (${eval_result})
        message(STATUS "\nAdding sample '${sample_name}'")
        add_subdirectory(${sample_name})
    else()
        message(STATUS "Skipping sample '${sample_name}'")
    endif()
endforeach()

