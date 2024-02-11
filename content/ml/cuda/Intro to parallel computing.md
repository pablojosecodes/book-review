---
title: Intro to parallel computing
---

Core issue: We need more resources / faster programs for modern applications


Traditional approach was **sequential** (think back to original Neumann design)
- Rely on advances in hardware- increased clock speed
- Get better microprocessors

But processor cores are no longer increaseing at the rate they were- which slows down the whole industry

The solution is **parallel programs**, which led to the concurrency revolution

# Heterogeneous Parallel Computing

Since 2003, 2 main trajectories for microprocessors
- **multicore**: maintain sequential program execution but move into multiple cores
- **many-thread**- focus on execution throughhput (e.g. GPUs)
	- Much more performant

Why the performance gap?
- GPUs oriented on throughput
- CPUs on latency (but consumes capacity which could otherwise be sent on more execution units / memory access)

> Note: reducing latency is much more expensive than increasing throughput. 

Why did GPUs win?
- Large existing user base (gaming)
- CUDA


# Why do we want more speed or parallelism?

- As we get higher quaity things, hard to go back to older tech (think HDTV)
- Better Uis
- Machine learning + research


# Speeding up real applications

**Speedup**: ratio of time used to excute system B over time to execute system A
- e.g. 200 seconds compared to original of 10, would be a 10x speedup

How to know what’s achievable?
- Percentage spent on paralelizable calculates upper limit (ie. can’t speed up the “peach flesh” portions)
- How fast data can be accessed from / written to the memory
- Original suitability of CPU to application 

# Why is Parallel Programming hard?

(1) Designing paralle algorithms with same level of compelxity as requentia agoirhtm is hard
- Non-intuitive
- Redudnat work potentially

(2) Speed of applications can be limited by memory access latency and/or throughput
- “Memory bound applications”- can optimize this

(3) Execution speed of parallel programs is more sensitive to input data characteristics than for sequential programs.
- Can use regularization techniques

(4) Some applications require threads to coaborate with each other
- Require using **synchronization operations**

Most of these have ben addressed by researchers

# Overarching goals / uses

(1) Goal: program massively parallel processors to achieve high perforamcne
- Intuition
- Knowledge of hardware

(2) Teach parallel programming for correct functionality and reliability
- Necessary if you want to support users

(3) scalability across future hardsware generations
- Have programs that can scale up to level of performance of new generations of machines


# Architecture tradeoffs



### CPU: Latency-Oriented Design
- **A few powerful ALUs (Arithmetic Logic Units):**
	- Capable of performing complex operations.
	- Designed to reduce the latency of each operation.
- **Large caches:**
	- To mitigate the latency of memory access by keeping data closer to the processing units.
	- Caches are optimized for quick access to data, reducing the time to retrieve information.
- **Sophisticated control:**
	- Includes mechanisms like branch prediction to anticipate the directions of branches (if/else conditions) and prepare execution paths.
	- Employs data forwarding to mitigate data hazards (delays caused by data not being ready when needed).

### GPU: Throughput-Oriented Design
- **Many small ALUs:**
	- Focused on performing many operations in parallel.
	- Trades off the speed of individual operations for the ability to do many simultaneously, prioritizing throughput over latency.
- **Small caches:**
	- Less cache per ALU compared to CPUs.
	- More silicon area is dedicated to ALUs rather than cache.
- **Simple control:**
	- Less complex control logic than CPUs.
	- More of the GPU's silicon area is allocated for computation rather than sophisticated control logic.

