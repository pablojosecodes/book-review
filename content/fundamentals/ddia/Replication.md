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

Until now, we’ve glossed over the actual implementation of leader-based replication logs. let’s look into the most common methods

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





