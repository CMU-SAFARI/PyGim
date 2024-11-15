cmake_minimum_required(VERSION 3.10)

#enable_testing()

get_filename_component(TEMPLATE_DPU_PROJECT_RELATIVE_PATH ${CMAKE_CURRENT_LIST_DIR}/dpu_kernels ABSOLUTE)


function(add_dpu_project name DPU_SRC_LIST
                              DPU_INCLUDE_DIRS
                              NR_TASKLETS
                              DPU_ARGN)

set(SOLVED_DPU_SRC_LIST "")
set(SOLVED_DPU_INCLUDE_DIRS "")

foreach (SRC ${DPU_SRC_LIST})
     get_filename_component(SOLVED_SRC ${SRC} ABSOLUTE)
     list(APPEND SOLVED_DPU_SRC_LIST ${SOLVED_SRC})
endforeach()
foreach (SRC ${DPU_INCLUDE_DIRS})
     get_filename_component(SOLVED_SRC ${SRC} ABSOLUTE)
     list(APPEND SOLVED_DPU_INCLUDE_DIRS ${SOLVED_SRC})
endforeach()


set(DPU_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/dpu_kernels)
set(DPU_BINARY_NAME ${name}_dpu)
if (NOT DEFINED UPMEM_HOME)
        if ( "$ENV{UPMEM_HOME}" STREQUAL "")
                set(UPMEM_HOME "/usr")
        else ()
                set(UPMEM_HOME $ENV{UPMEM_HOME})
        endif ()
endif ()
include(/usr/share/upmem/cmake/include/host/DpuHost.cmake)
link_directories("${DPU_HOST_LINK_DIRECTORIES}")

include(ExternalProject)

string(REPLACE ";" "|" SOLVED_DPU_SRC_LIST    "${SOLVED_DPU_SRC_LIST}")
string(REPLACE ";" "|" SOLVED_DPU_INCLUDE_DIRS    "${SOLVED_DPU_INCLUDE_DIRS}")

# pass ARGN to _dpu project
ExternalProject_Add(
        ${DPU_BINARY_NAME}
	    BINARY_DIR ${DPU_BINARY_DIR}
	    SOURCE_DIR ${TEMPLATE_DPU_PROJECT_RELATIVE_PATH}
        CMAKE_ARGS -DNAME=${DPU_BINARY_NAME}
                   -DDPU_SRC_LIST=${SOLVED_DPU_SRC_LIST}
                   -DDPU_INCLUDE_DIRS=${SOLVED_DPU_INCLUDE_DIRS}
                   -DCMAKE_TOOLCHAIN_FILE=${UPMEM_HOME}/share/upmem/cmake/dpu.cmake
                   -DUPMEM_HOME=${UPMEM_HOME}
                   -DNR_TASKLETS=${NR_TASKLETS}
			-DDPU_ARGN=${DPU_ARGN}
        BUILD_ALWAYS TRUE
        INSTALL_COMMAND ""
)

set(${name}-path  "${DPU_BINARY_DIR}/${DPU_BINARY_NAME}" PARENT_SCOPE)

endfunction()
