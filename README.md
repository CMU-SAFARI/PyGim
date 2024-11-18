# PyGim: An Efficient Graph Neural Network Library for Real Processing-In-Memory Architectures

This repository provides all the necessary files and instructions to reproduce the results of our [SIGMETRICS '25 paper](https://arxiv.org/abs/2402.16731).

> Christina Giannoula, Peiming Yang, Ivan Fernandez, Jiacheng Yang, Sankeerth Durvasula, Yu Xin Li, Mohammad Sadrosadati, Juan Gomez Luna, Onur Mutlu, Gennady Pekhimenko, "PyGim: An Efficient Graph Neural Network Library for Real Processing-In-Memory Architectures", SIGMETRICS'25. [Paper PDF](https://arxiv.org/abs/2402.16731.pdf)

[<i>PyGim</i>](https://arxiv.org/abs/2402.16731.pdf) is the first easy-to-use software framework to deploy Graph Neural Networks (GNNs) in real Processing-In-Memory (PIM) architectures. GNN execution involves both compute-intensive and memory-intensive kernels. PyGim integrates intelligent parallelization techniques for the memory-intensive kernels of GNNs tailored for real PIM systems, and provides an easy-to-use Python API for them. PyGim employs a cooperative GNN execution, in which the compute- and memory-intensive kernels are executed in processor-centric and memory-centric computing systems, respectively, to fully exploit the hardware capabilities. PyGim also integrates a lightweight autotuner that tunes the parallelization strategy of the memory-intensive kernel of GNNs based on the particular characteristics of the input graph, thus enabling high programming ease. 


## Cite PyGim

Please use the following citations to cite PyGim, if you find this repository useful:

Christina Giannoula, Peiming Yang, Ivan Fernandez, Jiacheng Yang, Sankeerth Durvasula, Yu Xin Li, Mohammad Sadrosadati, Juan Gomez Luna, Onur Mutlu, Gennady Pekhimenko, "[PyGim: An Efficient Graph Neural Network Library for Real Processing-In-Memory Architectures](https://arxiv.org/pdf/2402.16731.pdf)", arXiv:2402.16731 [cs.AR], 2024.


Bibtex entries for citation:
```
@article{Giannoula2025PyGimPomacs,
	author={Christina Giannoula and Peiming Yang and Ivan Fernandez Vega and Jiacheng Yang and Sankeerth Durvasula and Yu Xin Li and Mohammad Sadrosadati and Juan Gomez Luna and Onur Mutlu and Gennady Pekhimenko},
	title={PyGim: An Efficient Graph Neural Network Library for Real Processing-In-Memory Architectures}, 
	year = {2024},
	publisher = {Association for Computing Machinery},
	volume = {8},
	number = {3},
	url = {https://doi.org/10.1145/3700434},
	doi = {10.1145/3700434},
	journal = {Proc. ACM Meas. Anal. Comput. Syst.},
	articleno = {43},
}
```

```
@misc{Giannoula2025PyGimArXiv,
	  title={PyGim: An Efficient Graph Neural Network Library for Real Processing-In-Memory Architectures}, 
	  author={Christina Giannoula and Peiming Yang and Ivan Fernandez Vega and Jiacheng Yang and Sankeerth Durvasula and Yu Xin Li and Mohammad Sadrosadati and Juan Gomez Luna and Onur Mutlu and Gennady Pekhimenko},
      year={2024},
      eprint={2402.16731},
      archivePrefix={arXiv},
      primaryClass={cs.AR}
}
```

## Installation
NOTE: You need to run following command on the machine with UPMEM hardware and runtime library.

- Clone the project by
  ```Bash
  cd path/to/source
  git clone git@github.com:CMU-SAFARI/PyGim.git 
  ```
- Create conda virtual environments.
  ```Bash
  conda create -n=pygim python=3.8
  ```
- Install required Libs for PyGim
  ```Bash
  conda activate pygim
  cd path/to/source/PyGim/Libs
  bash ./install_libs.sh
  ```

- Build different backends
   ```Bash
  cd path/to/source/PyGim/backend_pim
  bash ./build.sh
  ``` ## Starter Example
- Run the SPMM test code:
  ```Bash
  cd path/to/source/PyGim
  mkdir ./data
  python3 ./spmm_test.py --datadir=./data --dataset=Reddit
  ```
- Run end-to-end inference:
  ```Bash
  cd path/to/source/PyGim
  python3 ./inference.py  --datadir=./data --dataset=Reddit
  ```

## Usage
- For spmm test(spmm_test.py), it has mutiple parameters:
    ```
    --dataset: select the graph data sets.
    --datadir: select the path to store the data sets and pre-processed data.
    --version: select backend version, must be in ["spmm", 'grande', "spmv", "cpu"].
    --tune: usage autotuner to overwrite the parallelization configation.
    --lib_path: path for PyGim backend Lib.
    --hidden_size: hidden_size of dense matrix.
    --data_type: data type for the matrix, select from ["INT8", "INT32", "INT16", "INT64", "FLT32", "DBL64"].
    --sp_format: format of sparse matrix, it select from ["CSR", "COO"].
    --sp_parts: number of partitions for sparse matrix.
    --ds_parts: number of partitions for dense matrix.
    --repeat: number of repeat round.
    ```
- For inference test(inference.py), it has similar parameters:

    ```
    --dataset: select the graph data sets.
    --datadir: select the path to store the data sets and pre-processed data.
    --model: model to select, support "gcn", "gin" and "sage" for now.
    --num_layers: number of layers of the model.
    --lr: learning rate
    --version: select backend version, must be in ["spmm", 'grande', "spmv", "cpu"].
    --tune: usage autotuner to overwrite the parallelization configation.
    --lib_path: path for PyGim backend Lib.
    --hidden_size: hidden_size of dense matrix.
    --data_type: data type for the matrix, select from ["INT8", "INT32", "INT16", "INT64", "FLT32", "DBL64"].
    --sp_format: format of sparse matrix, it select from ["CSR", "COO"].
    --sp_parts: number of partitions for sparse matrix.
    --ds_parts: number of partitions for dense matrix.
    --repeat: number of repeat round.
    ```
