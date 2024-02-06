---
title: Consistency and Consensus
---
Lots of things can go wrong in distributed systems.

How can we build fault-toolerant distributed systems, assuming all the problems we’ve discusssed can occur?

We can use a similar approach as with transactions- allow our application to use general purpose abstractions with useful guarantees.

# Consistency Guarantees

Most databases provide a guarantee of **eventual consistency**, which is honestly a pretty weak guarantee. 

Here, let’s explore some stronger consistency models and their tradeoffs.

A quick roadmap
- What is linearilizability (strongest commonly used consitency model)
- Ordering GUarantees
- How to atomically commit a distributed transaction
# Linearizability
**Linearizability**
- Synonyms: atomic consistency, strong consistency, immediate consistency, external consistency
- Core idea: give client the illusion there’s only one replica (ie never have separate ones transmit different data)
- Basically, its a recency guarantee

## What Makes a System Linearizable?
Every read must return the value set by the most recent write!

> Note: Serializability vs Linearizability
> Serializability: isolation prperty of transactions. Guarantees transactions behave as if they had been executed in some serial order (order can be different from the order in which transactions were actually run)
> Linearizability: recency guarantees on operations of a register (individual object). Doesn’t group opreations into transactions so it doesn’t prevent problems like write skew


## Relying on Lineraizability

When is linearizabilitty useful / essential?

#### Locking and leader election
Use case:  Single leader replciation- need one leader- how to make sure you elect just one leader? 
- **Distributed locking** with linearizable operations

#### Constains and uniquess guarantees
Use case: username and email address must uniquely identify one user and file storage service canno have same path and filename
- Similar to lock- need linearizability

#### Cross-channel timing dependencies
Use case: multiple comunication channels


## Implementing Linearizable Systems
How can we actually implement a system with linearizable semantics?

Basic idea of linearizability: “behave as if only one copy of the data and all operations are atomic.”

Naive approach: use one copy of the data

But when using replication, how amenable are different methods to linearizability? (TODO fill out bullets)
- Single leader replication: potentially
- Consensus algorithms: linearizable
- Multi-leader replciation: not
- Leaderess replication: probably not

### Linearizability and quorums
Seems like strict quorum reads / writes should be linearizable, but this is only really possible at the cost of reduced performacnce- reader must rperofrmm read repeir synchronously

## The cost of Linearizability

Useful to explore pros / cons of linearizability especially as only some replication methods can provide it
TODO

#### CAP Theorem
Any linearizable database has this problem- the tradeoff:
- If your application **requires** linearizability and some replicas are disconnected, the some replicas can’t process requests while they’re disconnected
- If your application does not require linearizability,c an be written in a way that each replcia can process requests independently. Can be avilable in the face of a network but without being linearizable

**Insight**: Applications which don’t require linearizability can be more tolerant of network problems

> Note: this is known as the CAP theorem

CAP
- **Originally**: broad rule of thumb
- Created shift
	- Before: focus on distibuted systems with liearizable semantics
	- Then: wider design space
- Now superseded by more precise results
#### Linearizability and network delays
Very few systems are actually linearizable in practice

Not even RAM onn modern ulti-core CPU is linearizable
- Each CPU core has its own memory cache and store buffer
- Memory access first to the cache by default- cahngs are asynchronously written out to main memory

Why make this tradeoff? Performance! Linearizability is slow
# Ordering Guarantees
Linearizability implies that operations are executed in a well-defined order

> Remember: 
> 	- Leader in single leader replication determines order of writes in the replication log
> 	- Serializability is all about ensuring tranactions behave as if executed in some sequential order
> 	- Clocks

## Ordering and Casuaility
One reason why ordering keeps coming up
- It helps preserve **causality**
TODO
Why do we care about causality?
- Imposes an ordering on events

# Distributed Transactions and Consensus
