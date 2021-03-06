include(CMakeParseArguments)

#
# Check directory against a list of files
#

function(util_find_dir VAR)
    cmake_parse_arguments(util_find_dir "" "DOC" "PATHS;FILES" ${ARGN})

    foreach(dir ${util_find_dir_PATHS})
        set(dir_ok TRUE)
        foreach(file ${util_find_dir_FILES})
            if(NOT EXISTS "${dir}/${file}")
                set(dir_ok FALSE)
                break()
            endif()
        endforeach()
        if(dir_ok)
            set("${VAR}" "${dir}" CACHE PATH "${util_find_dir_DOC}" FORCE)
        endif()
    endforeach()
endfunction()

function(check_and_set_env_var var_name result)
    if (DEFINED ENV{${var_name}})
        set(value $ENV{${var_name}})
        message(STATUS "${var_name} is set to ${value}")
        if(NOT EXISTS ${value})
            message(WARNING "File doesn't exist: ${file}!")
            SET(result OFF PARENT_SCOPE)
            return()
        endif()
        SET(${var_name} ${value} PARENT_SCOPE)
        SET(result ON PARENT_SCOPE)
    else()
        message(WARNING "Undefined ${var_name} environment variable!")
        SET(result OFF PARENT_SCOPE)
    endif()
endfunction()

function (Download from to fatal result err)
    if((NOT EXISTS "${to}"))
        message(STATUS "Downloading from ${from} to ${to} ...")
        file(DOWNLOAD ${from} ${to}
            TIMEOUT 3600
            LOG log
            STATUS status
            SHOW_PROGRESS)

        list(GET status 0 status_code)
        list(GET status 1 status_string)

        if(NOT status_code EQUAL 0)
            set(ERR_MSG "downloading '${from}' failed\n"
                        "status_code: ${status_code}\n"
                        "status_string: ${status_string}\n"
                        "log: ${log}\n")
            message(WARNING ${ERR_MSG})
            if (fatal)
                message(FATAL_ERROR "${ERR_MSG}")
            else()
                set(${result} "OFF" PARENT_SCOPE)
                set(${err} "${ERR_MSG}" PARENT_SCOPE)
                return()
            endif()
         endif()
    else()
    endif()
    set(${result} "ON" PARENT_SCOPE)

endfunction(Download)

function(get_workload_name net_name precision target_platform fuse batch_size name)
    string(TOLOWER ${target_platform} target_platform)
    SET(fuse_str "not_fused")
    if(fuse STREQUAL "1")
        SET(fuse_str "fused")
    endif()
    if(fuse STREQUAL "NONE")
        SET(fuse_str "hfused_none")
    endif()
    set(${name} ${net_name}_${precision}_${target_platform}_${fuse_str}_b${batch_size} PARENT_SCOPE)
endfunction()

function(fail_with_faked_main error)
    add_definitions(-DFAKED_MAIN)
    add_definitions(-DERR_MSG="${error}")
    message(WARNING ${error})
endfunction(fail_with_faked_main)

# Download Caffe model and topology files and setup environment vars to these files
function(DownloadCaffeFiles rv model_file_name model_path_var model_download_link topology_file_name topology_path_var topology_download_link)
    set(${rv} ON PARENT_SCOPE)

    check_and_set_env_var(${model_path_var} result)
    if(NOT result)
        SET(${model_path_var} "${CMAKE_CURRENT_BINARY_DIR}/models/${model_file_name}")
        if(NOT EXISTS "${${model_path_var}}")
            Download("${model_download_link}" "${${model_path_var}}" 0 result err)
            if(NOT result)
                set(ERR_MSG "[ERROR] ${SAMPLE_NAME} sample won't work, as downloading Caffe Model was failed with error during the build\\n")
                fail_with_faked_main(${ERR_MSG})
                set(${rv} OFF PARENT_SCOPE)
                return()
            endif()
        endif()
    endif()
    check_and_set_env_var(${topology_path_var} result)
    if(NOT result)
        SET(${topology_path_var} "${CMAKE_CURRENT_BINARY_DIR}/models/${topology_file_name}")
        if(NOT EXISTS "${${topology_path_var}}")
            Download("${topology_download_link}" "${${topology_path_var}}" 0 result err)
            if(NOT result)
                set(ERR_MSG "[ERROR] ${SAMPLE_NAME} sample won't work, as downloading Deploy Topology was failed with error during the build\\n")
                fail_with_faked_main(${ERR_MSG})
                set(${rv} OFF PARENT_SCOPE)
                return()
            endif()
        endif()
    endif()
    set(${model_path_var} ${${model_path_var}} PARENT_SCOPE)
    set(${topology_path_var} ${${topology_path_var}} PARENT_SCOPE)
endfunction(DownloadCaffeFiles)

# generate temporary file
macro(temp_name fname)
    if(${ARGC} GREATER 1)
        set(_base ${ARGV1})
    else(${ARGC} GREATER 1)
        set(_base ".cmake-tmp")
    endif(${ARGC} GREATER 1)
    set(_counter 0)
    while(EXISTS "${_base}${_counter}")
        math(EXPR _counter "${_counter} + 1")
    endwhile(EXISTS "${_base}${_counter}")
    set(${fname} "${_base}${_counter}")
endmacro(temp_name)

# macro to evaluate expression from particular string
macro(eval expr)
    temp_name(_fname)
    file(WRITE ${_fname} "if (${expr}) \nset(eval_result TRUE)\nelse()\nset(eval_result FALSE)\nendif()")
    include(${_fname})
    file(REMOVE ${_fname})
endmacro(eval)

function(append_tmp_to_ld_library_path paths)
    SET(LD_LIBRARY_PATH_INITIAL $ENV{LD_LIBRARY_PATH} CACHE INTERNAL "Initial value for LD_LIBRARY_PATH" FORCE)
    SET(ENV{LD_LIBRARY_PATH} ${paths}:$ENV{LD_LIBRARY_PATH})
endfunction()

function(reset_ld_library_path)
	SET(ENV{LD_LIBRARY_PATH} ${LD_LIBRARY_PATH_INITIAL})
endfunction()
