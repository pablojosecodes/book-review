---
title: Compute architecture and 4 scheduling
---
modern GPU architecture

SM: streaming multiprocessor
- Array of which composes CUDA-capable GPU
- Composition
	- Several CUDA Cores
	- Shared control logic
	- Shared memory
- **Block-level granularity**: may have multiple blocks, but each block lives only on one SM 
	- Guaraentees threads in same block are scheduled simulatneously


Synchronization
- Barrier synchronization: Threads in the same block can synchronize with `__syncthreads()`
	- Holds up program until each thread in block reaches that location
- The `__syncthreads()` MUST be executed by all thread in a block


Transparent scalability: blocks can execute at any order 
- Makes programs architecture agnostic

Timing of threads within each block
- Assume that threads can execute in any order (unless you use barrier synch- than can sync them up)

**Warp**: unit of thread scheduling- how many threads are executed at the same time
- Typically size of 32 

SIMD: single instruction multiple data model
- SMs folow this
- At any instant in time, one instruction is fetched + executed for all threads in the warp

Control divergence


TODO