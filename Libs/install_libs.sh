#!bash

install_libtorch(){
    version=${1:-"1.13.1"}
    wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${version}%2Bcpu.zip
    unzip ./libtorch-shared-with-deps-${version}+cpu.zip
    rm ./libtorch-shared-with-deps-${version}+cpu.zip
}

install_libtorch
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric==2.5.1 ogb matplotlib
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
