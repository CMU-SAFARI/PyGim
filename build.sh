#!bash

cd ./backend_pim/spmm_default
mkdir build && cd build
cmake ../ && make -j

cd ../../spmm_grande
mkdir build && cd build
cmake ../ && make -j

cd ../../spmm_multigroup
mkdir build && cd build
cmake ../ && make -j

cd ../../spmv_sparseP
mkdir build && cd build
cmake ../ && make -j
