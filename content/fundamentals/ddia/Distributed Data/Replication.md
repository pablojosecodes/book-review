---
title: Replication
---


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
	- This single **leader** is allowed to write to the database.
	- After writing, the leader sends the change to all its followers (via something called a *replication log*). Each replica updates its local data
- For reading from a client
	- Clients can read from the database by querying any of the replicas or leaders


### Synchronous vs Asynchronous replication

Asynchronous replication: leader sends message but doesn’t wait for response from replica.
Synchronous replication: leader waits for confirmation from replica.

It would clearly be quite the feat to have a fully synchronous system at a large scale- you’d have to wait for each and every node to report back success and would thus be fault-tolerant on each replica.

Thus, **semi-synchronous** replication in systems is more common. This is where you make one follower synchronous and switch the node you make synchronous if it fails. 


### Setting up new followers

How do you set up new followers to an data system? 

This isn’t as simple as it may seem- remember that the leader is constantly propagating write changes, so you can’t simply make a direct copy of it- the data is always in flux. And ideally, you could update without undergoing downtime to lock the disk.

The process typically looks like the following
1. Take a consistent snapshot at some specific point in time
2. Copy this snapshot to new follower node
3. Request all data since the point in time the the snapshot was taken
4. Sync the node by going through the data backlog

### Handling Node Outages

How does one achieve high availability with the leader-based replication system we’ve outlined? Let’s approach it in terms of the two kinds of node failure we might see.

**Follower failure**

In the case of a follower node failing or disconnecting to the network, it can perform what’s known as **catch-up recovery**. It should have a local disk where its logged all of its data changes.  Thus, it can just catch up on the data changes that have been transmitted by the leader since its outage.

**Leader failure**

When the leader node fails, we must (i) promote a new leader and (ii) switch the follower nodes to begin reading from that new leader. This process is called failover.

Typical automatic failover process
1. **Detect failure**: commonly, consider leader node dead if it doesn’t respond for some predetermined stretch of time.
2. **Choose new leader**: either chosen by an election process or by a *controller node* which is previously tasked with the decision.
3. **Reconfiguration**: ensure that client send their writes to this new leader and the follower nodes recognize the new leader.

There are many things that can go wrong here. To name just a few-
- What if former leader had writes which weren’t communicated to the new leader?
- Split brain: in some situations, you may end up with two leaders accepting writes. 
- What is the right timeout before declaring leader dead?


### Replication Logs Implementation

Until now, we’ve glossed over the actual implementation of leader-based replication logs. let’s look into the most common methods for replication logs

#### **Statement-Based Replication**
The most basic replication method. In it, the leader logs each write request (statement) as a log to its followers. (e.g. send `INSERT VALUE INTO ROW`).

Issues
- Non-deterministic functions (e.g. `NOW()`)
- Statements which must be executed in the same order
- Statements with non-deterministic side effects

Real-world usage: VoltDB (ensures safety by requiring deterministic transactions)

#### **Write-ahead log shipping**

Leader writes to a log containing bytes of all the writes to the database, before executing the writes.
- These writes are lower-level than statement-based replication, which makes it closely coupled to the storage engine

Real-world usage: PostgreSQL and ORACLE use it.

#### **Logical (row-based) log replication**

Use different log formats for replication (logical log) and for the storage engine internals.  This decoupling allows more easy backward compatibility.

Logical log- sequence of records describing writes at granularity of row. For each, the log contains…
- Inserted row: new values of all columns
- Deleted row: enough information to uniquely identify 
- Updated row: enough information to uniquely identify + new values


Real-world usage: MySQL binlog uses this approach

#### **Trigger-based Replication**
What happens if you want to do things like only replicate a subset of the data or replicate between database types?

You may need to move replication into the application layer! But how? With triggers!

**Triggers** are custom application code you’ve registered to automatically execute when a data change occurs in database system.

Real-world usage: [Databus](https://github.com/linkedin/databus) for Oracle and [Bucardo](https://bucardo.org/) for Postgres

## Problems with Replication Lag


Replication is essential for tolerating node failures, scalability, latency

Leader based replication requires all writes go througha single node (**read-scaling** architecture)
- Very attractive for read-heavy workloads
- Only realistic for asynchronous replication (can’t synchronously replicate to all followers)

The lag between replicas is known as replication lag
-  Can become a problem at a large enough lag and scale

## Reading your own writes

Issue: Many apps let users submit data and view submission
- What issues might be caused by replication lag?
- Well, you might read stale data that you just wrote

Fix?
- **Read-after-write** consistency- guarantee that they’ll see updates they themselves submitted
- How? A few techniques
	- If reading something user may hae modified,read from the leader (e.g. read user’s own profile from leader)
	- Use other criteria to decide whether to read fromthe leader- could track replication lag or how recent submission was
	- Cient can remember timestamp of most recent write and ensure that replica has been served writes up to that point in time

But what about multiple devices? Harder
- No guarantee connections will route to same datacenter
- How to see timestamps across devices?

## Monotonic Reads
Issue: user seeing things moving back in time
- When you read from replica that is further forward in time and then aftre a replica that still hasn’t received updaes

How to fix? **monotonic reads**
- Basically- guarantee that reads in sequence won’t go backwards in time
- Sample implementation: make user always make read from same replica

## Consistent prefix Reads

Issue: causality broken
- e.g. messages routed in lag and answer appears before question

Sample solution: Make sure writes that are causally related to each other written to the same partition

## Solutions for Replication Lag

If you’re working with an eventually consistent system, ask “how will this behave if replication lag increases”

# Multi-Leader Replication

Big con with leader-based relication: only one leader and all writes need to go through it.

Multi-leader: more than one node can accept writes
- Master-master replication
- Active-active replication

## Use cases
Not useful within a single datacenter, but here are some useful situations
- Multi-Datacenter Operation
	- Leader in each datacenter
	- Higher performance, higher tolerance of outages/network problems
- Clients with Offline Operation
	- Local database which acts as leader
- Collaborative Editing
	- See ‘automatic conflict resolution’ in the book TODO


## Handling Write Conflicts

But now we ned conflict resolution mechanisms

**Conflict avoidance**
Simplest (and quite common) strategy for dealing with conflicts is to avoid them

But how? Well you could make all writes for a specific record go through the same leader.

**Converging towards a consistent database**

We know that in single leader databases, the writes are applied in a sequential order

But we don’t have a defined ordering of writes in a multi-leader configuration
- What should the final value be?

We can’t have each replica apply writes in the order that they see them- the conflict must be resolved in a **convergent** way

A couple methods of convergent conflict resolutoin
- Each write is given a unique ID- pick write with highest ID as winner (LWW)
	- Prone to data loss
- Each write is given a unique ID, and writes which originate at higher-numbered replica take precedence
- Merge the values somehow
- Record the conflict explicitly and have application code resolve the conflict later on

**Custom conflict resolution logic**

Resolving conflict may be app dependent

Your code can be executed either on write or on read
- On write: as soon as database detects conflictin log of replicated changes
- On read: When a conflict is detected, all the conflicting writes are stored. The next time the data is read, these multiple versions of the data are returned to the application- which deals with it.

**What is a conflict?**
Obvious- 2 writes concurrently

Also- e.g. meeting booking system where mutliple book same room



# Multi-leader Replication Topologies

**Replicaton topology**: communicatoin paths which writes take from one node to another
- Most general: all-to-all
	- Each leader sends writes to all other leaders
- Circular topology
	- Node receives writes from one node and forwards those writes to another node
- Star Topology
	- One designated root node forwards writes to all the other nodes

# Leaderless Replication

Replication so far has been based on the idea that a client sends a write rquest to one node and the database copies write to other replicas

But what about leaderless replication- accept writes directly from clients

A key distinction between different leaderless replciation methods
- Does the client write directly to several replcis or does a coordinator node do this on behalf of it?

## Writing to database when node is down
Failover doesn’t exist when you’re using leaderless replication
- User sends request to several nodes in parallel
- But how does node catch up after failing and coming back? 2 mechanisms
	- Read repair: If client detects stale data (ie 3/4 say something different than 1/4 nodes), writes right data back
	- Anti-entropy process: background process which looks for differences in data
- Quorum for reading and writing: the number of nodes needed to consider succesful writing/reading

## Limitations of Quorum Consistency
Even with conservative parameters, there are likely edge cases where stale values are returned
- These are pretty easy to think of- some unfotunae failure cases in timing and fails

In leaderless replication- how to **monitor for staleness**?


## Sloppy Quorums and Hinted Handoffs

Databases with well set quorums can tolerate the failure of individual nodes without needing failover

But- quourms are not as fault-tolernant as they could be
- Network interruption could cut off client from large number of database nodes

How to address this? **Sloppy quorum** would be the practice of still accepting writes and reads, even if you switch nodes from original “home” nodes- 
- **Hinted handoff**: after network comesonline- writes which were accepted temporarily by non-home nodes are sent to appropriate home nodes


## Detecting Concurrent Writes
Even with strict quorums, some databaes allow several clients to concurrently write to the same key- conflict will occur even with strict quorums

In order to become eventually consistent, replicas should converge to the same value.
- Databases aren’t great at this currently

Important to know all the failure cases and how to handle this on the application side!

Lots potential solutions here.
