cmake_minimum_required(VERSION 3.12)
project(backend_pim C CXX)
get_filename_component(Torch_DIR ../Libs/libtorch//share/cmake/Torch/ ABSOLUTE)


#include(${UPMEM_HOME}/share/upmem/cmake/include/host/DpuHost.cmake)
include(/usr/share/upmem/cmake/include/host/DpuHost.cmake)
include(dpu.cmake)
find_package(Torch REQUIRED)
set(CMAKE_BUILD_TYPE "Release")



IF(NOT DEFINED extra_def)
    set(extra_def "-DINT32=1 -DBLNC_NNZ=0 -DBLNC_NNZ_RGRN=1 -DBLNC_ROW=0")
    set(extra_def "${extra_def} -DBLNC_TSKLT_ROW=0 -DBLNC_TSKLT_NNZ_RGRN=0 -DBLNC_TSKLT_NNZ=1")
    set(extra_def "${extra_def} -DLOCKFREEV2=1 -DCG_LOCK=0")
    set(extra_def "${extra_def} -DROW_MERGE=0 -DBLOCK_MERGE=1")
    set(extra_def "${extra_def} -DSYNC=0")
ENDIF()



IF(NOT DEFINED NR_TASKLETS)
        set(NR_TASKLETS 16)
ENDIF()


IF(NOT DEFINED KERNEL)
        set(KERNEL "MUL_CSR")
ENDIF()

IF(NOT DEFINED PIM_SEQREAD_CACHE_SIZE)
    set(PIM_SEQREAD_CACHE_SIZE 32)
#    message(FATAL_ERROR "PIM_SEQREAD_CACHE_SIZE is not defined")
ENDIF()

set(extra_def "${extra_def} -DPIM_SEQREAD_CACHE_SIZE=${PIM_SEQREAD_CACHE_SIZE}")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 ${extra_def}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 ${extra_def}")
set(CPP_SRCS pytorch_api.cpp ops.hpp utils.hpp)
set(C_SRCS spmm_mul_csr.c support/timer.c support/partition.c)
add_library(backend_pim SHARED ${CPP_SRCS} ${C_SRCS})


target_include_directories(backend_pim PUBLIC . "/usr/include/dpu/")
#target_include_directories(backend_pim PUBLIC . "${UPMEM_HOME}/include/dpu/")
#target_link_directories(backend_pim PRIVATE ${UPMEM_HOME}/lib/)
target_link_libraries(backend_pim "${TORCH_LIBRARIES}")
target_link_libraries(backend_pim -ldpu -ldpuverbose)

target_compile_definitions(backend_pim PUBLIC -DNR_TASKLETS=${NR_TASKLETS})

if(KERNEL MATCHES "MUL_CSR")
        message("add MUL_CSR kernel")
        add_dpu_project(spmm_mul_csr "dpu_kernels/spmm_grande_mul_csr_dpu.c" "./support" ${NR_TASKLETS} ${extra_def})
        target_compile_definitions(backend_pim PUBLIC -DSPMM_KERNEL_BINARY="${spmm_mul_csr-path}")
#elseif(KERNEL MATCHES "MUL_COO")
#        add_dpu_project(spmm_mul_coo "dpu_kernels/spmm_mul_coo_dpu.c" "./support" ${NR_TASKLETS} ${extra_def})
#        target_compile_definitions(backend_pim PUBLIC -DSPMM_KERNEL_BINARY="${spmm_mul_coo-path}")
endif()





