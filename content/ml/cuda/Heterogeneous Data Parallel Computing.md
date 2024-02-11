---
title: Heterogeneous Data Parallel Computing
---

Data paralelism: computation work on different parts of dataset done independently/in parallel

Task parallelism: task decomposition (ie. vector addition + matrix-vector application independently)


Slow programs- issue is usually too much data to process



# CUDA C Structure

Terms
- Host: CPU
	- `__host__` keyword- function is a CUDA host function 9only executed/called on host
- Device: GPU
- Host code: CPU serial code
- Grid: all threads launched on a device to execute the kernel
- Block: each of same size on grid, contains threads
- Host memory: 
	- `_h` indicates object in device memory?
	- `__device__` keyword- function is a CUDA device function (can be called from only kernel or device function)
- Global memory: 
	- `_d` indicates object in device global memory
	- `__global__` keyword- function is a CUDA C kernel function
- Configuration parameters: given between `<<<` and `>>>`
	- First: number of blocks in grid
	- Second: number of threads in block

Functions
- `cudaMalloc()`- call on host code to allocate global memory for an object
	- Allocates object in device global memory
	- Parameters
		- Address of pointer to allocated object
		- Size of allocated object in term of bytes
- `cudaFree`-
	- Frees object from device global memory
	- Parameter
		- Pointer to freed object
- `cudaMemcpy`
	- Memory data transfer
	- Parameters
		- Pointer to destination
		- Pointer to source
		- Number of bytes copied
		- Type/direction of transfer (device/host → device/host)
- `cudaMemcpyHostToDevice` and `cudaMemcpyDeviceToHost`
	- Predefined constants of the CUDA programming environment

Within Kernel
- `blockDim`
	- `.x` (if 1D)- indicates the total number of threads in each block
- `threadIdx`
	- `.x` (if 1D)- current thread within block
- `blockIdx`
	- `.x` (if 1D)- current block coordinate


General structure
1. Alocate GPU memory
2. Copy data to GPU memory
3. Perform computation on GPU
4. Copy data from GPU memory
5. Deallocate GPU memory


How to compile C kernels?
- NVCC
- C with CUDA turns into
	- Host code (straight ANSI)
	- Device code (PTX)- executed on CUDA-capable GPU device




Example run through
```c++
#include "timer.h"

void vecadd_cpu(float *x, float *y, float *z, int N) {
    for(unsigned int i = 0; i < N; ++i) {
        z[i] = x[i] + y[i];
    }
}

void vecadd_gpu(float *x, float *y, float *z, int N) {
    // Allocate GPU memory
    float *x_d, *y_d, *z_d;
    cudaMalloc((void**)&x_d, N*sizeof(float));
    cudaMalloc((void**)&y_d, N*sizeof(float));
    cudaMalloc((void**)&z_d, N*sizeof(float));

    // Copy to the GPU
    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Call a GPU kernel function (launch a grid of threads)
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = N/512;
    vecadd_kernel<<<numBlocks, numThreadsPerBlock>>>(x_d, y_d, z_d, N);

    // Copy from the GPU
    cudaMemcpy(z, z_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate GPU memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}
```

# Multi-dimensional grids and data

We’ve talked about one-dimensional grids of threads, but how about multimimensional arrays of data?

> Remember the built-in block and thread variables.


In general, grid is a 3D array of bocks nad each block is a 3D array of threads

As an example (for creating 1D grid with 32 blocks with 128 threads)
```c
dim3 dimGrid(32,1,1);
dim3 dimBlock(128,1,1);
vecAddKernel <<<dimGrid, dimBlock>>>(..)
```
ETC.
