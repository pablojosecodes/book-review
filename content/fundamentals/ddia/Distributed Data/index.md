---
title: Distributed Data
---
While the first section of this reference material focused on learning about data systems on a single machine, let’s now focus on `what happens to data systems when they need to incorporate multiple machines?`

This (scaling to multiple machines) is necessary for
- **Scalability**- spreading the load
- **Fault tolerance**- multiple machines = redudancy
- **Latency**- spreading the servers physically

## Scaling to Higher Load

Let’s think from first principles: how might you approach the issue of needing to scale to handle higher load?

The most naive solution would be what’s known as **vertical scaling**- variations of buying a more powerful machine. You could opt for a…
- Shared memory architecture: join many CPUs/RAM chips/disks into one OS
	- Issue: gets very expensive very fast, not super efficient
- Shared-disk architecture: independent CPUs/RAM but share array of disk for storing data
	- Issue: limited by overhead of locking

But what’s become common are **shared-nothing architectures (ie. horizontal scaling)**
- In this architecture, each machine has completely independent CPU/RAM/disk and only coordinates with other machines (known as *nodes*) using a conventional network
- Benefits
	- Use whatever machines you prefer
	- Very distributable

But how do you distribute data across multiple nodes? 

There are two main approaches (which are often used together)
- Replication: same data on different nodes
- Partitioning: big dataset into smaller subsets

The following sections will mostly deal with distributed shared-nothing architectures.


> [!TOC]
> 1. [[Replication|Replication]] 
> 2. [[Partitioning|Partitioning]] 
> 3. [[Transactions]]
> 4. [[The Trouble with Distributed Systems]]
> 5. [[Consistency and Consensus]]