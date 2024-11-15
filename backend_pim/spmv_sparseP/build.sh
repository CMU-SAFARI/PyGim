#!bash

ROOT_DIR=`dirname $(readlink -f $0)`
CURRENT_DIR=`pwd`
echo "${ROOT_DIR}"
UPMEM_HOME="${ROOT_DIR}/../Libs/UPMEM_SDK"
LIBTORCH_HOME="${ROOT_DIR}/../Libs/libtorch"
source ${UPMEM_HOME}/upmem_env.sh


cd ${ROOT_DIR}
if [[ -d "build" ]]; then
    rm -rf ./build
fi
mkdir build 
cd build
pwd
cmake ${ROOT_DIR}
make -j
cd ${CURRENT_DIR}
export DPU_KERNEL_DIR="${CURRENT_DIR}/build/dpu-kernels"



