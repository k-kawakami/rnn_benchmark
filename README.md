# rnn_benchmark

This is a GPU speed test with standard language model.
All the experiments are done with Pytorch & cudnn backend.

```
git clone https://github.com/k-kawakami/rnn_benchmark.git
cd rnn_benchmark
mkdir data
bash download.sh
CUDA_VISIBLE_DEVICES=0 python run.py -rnn LSTM -nlayers 1 -emb_dim 1024 -hid_dim 1024 -tied 0 -epochs 10 -optimizer Adam -lr 0.002 -dropout 0.5 -batch_size 128 -seq_len 128
```

## Evaluation

The evaluations are done on a standard language modeling dataset, [Wikitext-2](https://metamind.io/research/the-wikitext-long-term-dependency-language-modeling-dataset/).

## Model Configuration

Since we are not trying to establish SoTA results, we used large minibatch size to consume large amount of GPU memory. 
The hyperparameters we used are the following.

```
-rnn LSTM 
-nlayers 1 
-emb_dim 1024 
-hid_dim 1024 
-tied 0 
-epochs 100 
-optimizer Adam 
-lr 0.002 
-dropout 0.5 
-batch_size 128 
-seq_len 128
```

## Results

We ran the model for 10 epochs and report averaged number of processed words per second (wps).

- Machine: train wps / test wps [words/sec]
- Tesla P100-SXM2-16GB (DGX1): Train 547,106.703 wps / Eval: 3,365,052.456
- Tesla P100-PCIE-16GB: Train 497,944.387 wps / Eval: 3,692,290.684
- GeForce GTX TITAN X: Train 354,534.268 wps / Eval: 4,399,055.094

### Details

- DGX1
```
Tesla P100-SXM2-16GB, 375.20, 16308 MiB, 0 MiB, 16308 MiB

Python VERSION: 2.7.13 |Anaconda 4.3.1 (64-bit)| (default, Dec 20 2016, 23:09:15) 
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
pyTorch VERSION: 0.1.12_2
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Sun_Sep__4_22:14:01_CDT_2016
Cuda compilation tools, release 8.0, V8.0.44
CUDA VERSION: 0
CUDNN VERSION: 6021
Number CUDA Devices: 1
Active CUDA Device: GPU: 0


```

- P100
```
Tesla P100-PCIE-16GB, 367.48, 16276 MiB, 0 MiB, 16276 MiB

Python VERSION: 2.7.13 |Anaconda 4.3.0 (64-bit)| (default, Dec 20 2016, 23:09:15) 
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
pyTorch VERSION: 0.1.12_2
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2016 NVIDIA Corporation
Built on Sun_Sep__4_22:14:01_CDT_2016
Cuda compilation tools, release 8.0, V8.0.44
CUDA VERSION: 0
CUDNN VERSION: 6021
Number CUDA Devices: 1
Active CUDA Device: GPU: 0

CPU INFO : Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
```

- Titan X
```
GeForce GTX TITAN X, 352.79, 12287 MiB, 23 MiB, 12264 MiB

Python VERSION: 2.7.13 |Anaconda 4.3.1 (64-bit)| (default, Dec 20 2016, 23:09:15) 
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)]
pyTorch VERSION: 0.1.12_2
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2015 NVIDIA Corporation
Built on Tue_Aug_11_14:27:32_CDT_2015
Cuda compilation tools, release 7.5, V7.5.17
CUDA VERSION: 0
CUDNN VERSION: 6021
Number CUDA Devices: 1
Active CUDA Device: GPU: 0
Loaded 2075677 words from d
```

- Titan X (Mulga)
```

```
