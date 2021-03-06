# rnn_benchmark

This is a GPU speed test with language modeling.

```
git clone https://github.com/k-kawakami/rnn_benchmark.git
cd rnn_benchmark
mkdir data
bash download.sh
CUDA_VISIBLE_DEVICES=0 python run.py -rnn LSTM -nlayers 1 -emb_dim 1024 -hid_dim 1024 -tied 0 -epochs 10 -optimizer Adam -lr 0.0002 -dropout 0.5 -batch_size 128 -seq_len 128 -clip 0.1 -seed 1234 -cudnn
```

## For contributors!

Thank you very much for running experiments!

Please provide deviceQuery information and cpuinfo of your machine below.

``` operating system
cat /proc/version 
```

```  scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
```

``` cupinfo
cat /proc/cpuinfo
```

``` deviceQuery
/usr/local/cuda-8.0/samples/1_Utilities/deviceQuery/deviceQuery
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
-epochs 10
-optimizer Adam
-lr 0.0002
-dropout 0.5
-batch_size 128
-seq_len 128
-clip 0.1
-seed 1234
-cudnn
```

To achieve the state-of-the-art performance, you need to tune parameters. 

For example, this parameter will give 94.614 on test set.

```
python run.py -rnn LSTM -nlayers 1 -emb_dim 1024 -hid_dim 1024 -tied 1 -epochs 1000 -optimizer Adam -lr 0.0002 -dropout 0.5 -batch_size 20 -seq_len 32 -clip 0.1 -seed 100
```

## Results

### Single layer LSTM with 1024 hidden units

We ran the model for 10 epochs and report the number of processed words per second (wps).

| With CuDNN                  |   Train   | Test (no backprop) |
|-----------------------------|:---------:|:------------------:|
| Tesla P100-SXM2-16GB (DGX1) | 78386.382 |     3111193.054    |
| Tesla P100-PCIE-16GB        | 72493.573 |     3464339.564    |
| GeForce GTX TITAN X         | 43774.577 |     4225782.301    |


| Without CuDNN               |   Train   | Test (no backprop) |
|-----------------------------|:---------:|:------------------:|
| Tesla P100-SXM2-16GB (DGX1) | 27508.843 |     266949.819     |
| Tesla P100-PCIE-16GB        | 25386.440 |     246275.124     |
| GeForce GTX TITAN X         | 15918.442 |     266936.834     |

### 2-layer Deep LSTM with 512 hidden units for each layer

CuDNN compute multi-layer LSTM in parallel.

We expected to get more improvements with cudnn on deep lstm case than single lstm case but that's not ture.

Computations in embedding and softmax might have different effects.


| With CuDNN                  |   Train   | Test (no backprop) |
|-----------------------------|:---------:|:------------------:|
| Tesla P100-SXM2-16GB (DGX1) | 53824.334 |     1778586.693    |
| Tesla P100-PCIE-16GB        | 49397.225 |     1875395.719    |
| GeForce GTX TITAN X         | 32534.504 |     2559483.339    |


| Without CuDNN               |   Train   | Test (no backprop) |
|-----------------------------|:---------:|:------------------:|
| Tesla P100-SXM2-16GB (DGX1) | 37693.830 |     326144.706     |
| Tesla P100-PCIE-16GB        | 34889.013 |     299286.352     |
| GeForce GTX TITAN X         | 22443.046 |     332199.661     |


## Details

- DGX1
```
OS: Linux version 4.4.0-45-generic (buildd@lcy01-08) (gcc version 4.8.4 (Ubuntu 4.8.4-2ubuntu1~14.04.3) ) #66~14.04.1-Ubuntu SMP Wed Oct 19 15:05:38 UTC 2016

Scaling Governor: performance

GPU: Tesla P100-SXM2-16GB, 375.20, 16308 MiB
Device 7: "Tesla P100-SXM2-16GB"
  CUDA Driver Version / Runtime Version          8.0 / 8.0
  CUDA Capability Major/Minor version number:    6.0
  Total amount of global memory:                 16308 MBytes (17100439552 bytes)
  (56) Multiprocessors, ( 64) CUDA Cores/MP:     3584 CUDA Cores
  GPU Max Clock rate:                            1481 MHz (1.48 GHz)
  Memory Clock rate:                             715 Mhz
  Memory Bus Width:                              4096-bit
  L2 Cache Size:                                 4194304 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 5 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 138 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

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

CPU: Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
```

- P100
```
OS: Linux version 3.10.0-514.el7.x86_64 (builder@kbuilder.dev.centos.org) (gcc version 4.8.5 20150623 (Red Hat 4.8.5-11) (GCC) ) #1 SMP Tue Nov 22 16:42:41 UTC 2016

Scaling Governor: No

GPU: Tesla P100-PCIE-16GB, 367.48, 16276 MiB
Device 1: "Tesla P100-PCIE-16GB"
  CUDA Driver Version / Runtime Version          8.0 / 8.0
  CUDA Capability Major/Minor version number:    6.0
  Total amount of global memory:                 16276 MBytes (17066885120 bytes)
  (56) Multiprocessors, ( 64) CUDA Cores/MP:     3584 CUDA Cores
  GPU Max Clock rate:                            405 MHz (0.41 GHz)
  Memory Clock rate:                             715 Mhz
  Memory Bus Width:                              4096-bit
  L2 Cache Size:                                 4194304 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Enabled
  Device supports Unified Addressing (UVA):      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 132 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

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

CPU: Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
```

- Titan X
```
OS: Linux version 4.2.0-27-generic (buildd@lcy01-23) (gcc version 4.8.2 (Ubuntu 4.8.2-19ubuntu1) ) #32~14.04.1-Ubuntu SMP Fri Jan 22 15:32:26 UTC 2016

Scaling Governor: powersave

GPU: GeForce GTX TITAN X, 352.79, 12287 MiB
Device 0: "GeForce GTX TITAN X"
  CUDA Driver Version / Runtime Version          7.5 / 7.5
  CUDA Capability Major/Minor version number:    5.2
  Total amount of global memory:                 12287 MBytes (12884180992 bytes)
  (24) Multiprocessors, (128) CUDA Cores/MP:     3072 CUDA Cores
  GPU Max Clock rate:                            1240 MHz (1.24 GHz)
  Memory Clock rate:                             3505 Mhz
  Memory Bus Width:                              384-bit
  L2 Cache Size:                                 3145728 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 3 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

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

CPU: Intel(R) Xeon(R) CPU E5-2630 v3 @ 2.40GHz
```

## Acknowledgements

Thanks [Guillaume Lample](https://github.com/glample) and [Sandeep Subramanian](https://github.com/MaximumEntropy) for valuable comments! They also have nice rnn-benchmarks.
