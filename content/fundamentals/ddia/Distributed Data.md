---
title: Distributed Data
---
While the first section of this reference material focused on learning about data systems on a single machine, let’s now focus on `what happens to data systems when they need to incorporate multiple machines?`

This (scaling to multiple machines) is necessary for
- Scalability- spreading the load
- Fault tolerance- multiple machines = redudancy
- Latency- spreading the servers physically

## Scaling to Higher Load

How might you approach the issue of needing to scale to handle higher load?

The most naive solution would be what’s known as **vertical scaling**- variations of buying a more powerful machine.

Subsets of this would be..
- Shared memory architecture: join many CPUs/RAM chips/disks into one OS
	- Issue: gets very expensive very fast, not super efficiecnt
- Shared-disk architecture: independent CPUs/RAM but share array of disk for storing data
	- Issue: limited by overhead of locking

Instead, what’s become common are **shared-nothing architectures (ie. horizontal scaling)**
In this architecture, each machine has completely independent CPU/RAM/disk and only coordinates with other machines (known as *nodes*) using a conventional network
Some Benefits
- Use whatever machines you prefer
- Very distributable

But how do you distribute data across multiple nodes? There are two main approaches (which are often used together)
- Replication: same data on different nodes
- Partitioning: big dataset into smaller subsets

The following section will mostly deal with distributed shared-nothing architectures.

# Replication

Recall, Replication = keeping copy of same data on multiple machines conneted via a network
- Keeps data close to your users
- Increases availability
- Increases read throughput

Note: for purposes of this section on `replication` we’ll assume your dataset is small enough for each machine to hold a copy of the entire dataset

Where might issues arise with keeping multiple copies of the same data in a distributed fashion?  Well, the hard part of course is handling the changes to the replicated data. There are 3 common algorithms for dealing with this between nodes, which we’ll look into now
- Single-leader
- Multi-leader
- Leaderless

## Single Leader 

The general process for a single leader architecture is as follows:
- For writing from a client
	- Writes must be sent to the leader
	- This single **leader** is allowed to write to database.
	- After writing, the leader sends the change to all its followers (via something called a *replication log*). Each replica updates its local data
- For reading from a client
	- Clients can read from the database by querying any of the replicas or leaders





