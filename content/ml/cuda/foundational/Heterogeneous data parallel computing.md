---
title: Heterogeneous data parallel computing
---

Core issue: We need more resources / faster programs for modern applications


Traditional approach: sequentia (think back to original Neumann design)
- Rely on advances in hardware- increased clock speed
- Get better microprocessors

But~ processor cores are no longer increaseing at the rate they were- which slows down the whole industry

Solution- **parallel programs**- the concurrency revolution

# Heterogeneous Parallel Computing

Since 2003, 2 main trajectories for microprocessors
- **multicore**: maintain sequential program execution but move into multiple cores
- **many-thread**- focus on execution throughhput (e.g. GPUs)
	- Much more performant

Why the performance gap?
- GPUs oriented on throughput, CPUs on latency (but consumes capacity which could otherwise be sent on more execution units / memory access)



> Note: reducing latency is much more expensive than increasing throughput. 

Other reasons to use GPUs/
- User base
- CUDA


# Why more speed or parallelism?

