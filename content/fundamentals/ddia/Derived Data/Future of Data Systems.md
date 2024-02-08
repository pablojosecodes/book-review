---
title: Future of Data Systems
---
We’ve discussed how things are, but how should they be? (when it comes to desining applications)

# Data Integration

We know that there is no one right solution to broad problems like storing data.

The first chalenge is to figure out which software products are useufl where.

Next challenge is that data is often used in different ways- so you likely cant just use one software product

## Combining Specialized Tools by Deriving Data

Common example: needing to integrate OLTP databse with full-text search.
- What should you use?
- Some databases includ full-text indexing, but more sophisticated search needs specialist information retreival tools
- Maybe you need database, serach index, and copies of data in analytics systems. 
- Mayve you also need caches or denormalized versions of objects.

**Reasoning about dataflows**

When you need copies of some data to be maintained for some access patterns.

How to get data into the riht places in the fright. formats?


**Derived data vs. distributed transactions**

Classic approahc for keeping data systems consistent involves distributed transactions

How does derived data systems compare?

- Abstract level: similar goal with different means
- biggest difference: transaction systems usually proide linearizability, while derived data as ystems often update asynchronously

**Limits of total ordering**

Smaller systems can handle totally ordered event log, but larger/complex workloads cause issues
- Usually need all events ot pass througha. single leader node
- If geographical distribution, typically have a separate leader in each datacenter (undefined ordering of events in two diffeernt datacenters)
- When applications deployed as microservices, common design choice- deploy each service and state as independent unit
	- No defined order for events from different services
- Some applications maintain client-side state updated on user input and work offline
	- Client and server likely to see different ordering of events


**Ordering events to capture causality**
Lack of total order is not a brpoblem when there is no causal link between events

Causal dependencies can arise in subtle ways
- Example: user remoces friend, insults them- storing friendship status in one place and message in another palce → ordering dependency between unfriend event and message send event may be lost

Simple answers to the problem
- Logical timestamps 
- Conflict resolution algorithms

## Batch and stream processing
Goal of data integration = data in right form in all the right places.

Batch and stream processing help with this goal.

**Maintaining derived state**
Batch processing has a strong functional flavor (encourages deerinsitic pure functions with outputs that have no side effects)
- Simplifies reasoning about dataflows
- Good for fault tolerance

Derived data systems could be maintained synchronously, but the asynchrony is what makes systems based on event logs robust.


**Reprocessing data for Aplication Evolution**
Batch processing allows reprocessing to get new views.

Reprocessing is a good mechanism for mainmaintainingting a system
- Allows for more advanced schema evolution

Derived views allow gradual evolution
- Can maintain old and new schema side by side as two independenlty derived views onto the same underlying data

**Lambda architecture**
We know
- Batch processing: reprocess historical data
- Stream processing: process recent data

How to combine the two? **Lambda architecture** is a proposal

Core idea: incoming data recorded by appending mimmutable events to always-growing dataset

Derive read-optimized events from these events
- Using both batch processing and stream processing in parallel
- Stream processor: quickly produces approimeate update to view
- Batch processor: consumes same set of events and creates correct version of the derived view

Number of practical problems: Need to-
- Maintain same logic for running both in batch/stream
- Merge steram/batch

**Unifying batch and stream process**
Requires the following features (which are becoming increasingly available)
- Ability to replay historical events through same processing engine which handles stream of recent events
- Exactly-once semantics for stream processors (ensure output is same as if no faults occurred)
	- Tools for windowing by event time, not processing time
	- Processing time is meaningfless when reprocessing historical events

# Unbundling Databases

Abstractly, databases, Hadoop, OSs all do the same things
- Store data
- Process/query the data

Comparison
- Databases: some data model
- Hadoop: something like a distributed verson of Unix
- OS: data in files

Different philosophies
- Unix: logical but low-level hardware abstraction
	- Pipes
	- Files
- Relational database: high level abstraction
	- SQL 
	- Transactions

We shouild try to combine the best of both worlds.

## Composing Data Storage Technologies
We’ve talked about lots of feautres from databasees and hwo they work. 

There are parallels between features in databases and derived data systems

**Creating an index**
When you CREATE INDEX
- Db scans over consistent snapshot
- Picks out fied values indexed
- Sorts
- Writes out index
- Process backlog of writes
- Then- keep index up to date with table’s writes

Similar to setting up follower replica and to bootstrapping change data capture in a streaming system

**Meta-Database of Everything**

Dataflow across org starts looking like one huge database

Whenever a batch, steram, or ETL process trasports data from one place/form to another place/form, it is acting like database subsystem that keepds indexes or materialized views up to date

In this view
- Batch/steram processors = implementation s of triggers, stored procedures, materialized view maintenance routines
- Derived data systems: different index types

How to compose different storage and processing tools (if we accept that no single data model / storage format can be suitable for all access pattens)? Two different avenues-
1. **Federated databases**: unifying reads
	1. Create a unified query interface to a wide variety of underlying storage engies and processing methods
	2. Examples: PostgreSQL’s foreign data wrapper
2. **Unbundled databases**: unifying writes
	1. When composing several storage systems, need to ensure data changes end up in alt the right places

**Making unbundling work**
Federation + unbundling have same goal:
- Goal: compose a reliable, scalable, maintainable system out of diverse components

Federated read-only querying is just about mapping one data model to another

But unbundling- how to keep writes to several storage systems in sync?

Traditional (wrong) approach for synchrnizing writes
- Distributed transactions across heterogeneous sotorage systems

More robust solution for when data crosses boundary between different techonogies: asynchronous event log with idempotent writes

Log based integration’s big advantage = loose coupling
1. System lvel: more robust to outages- even tlog can buffer messages
2. Human level: can work on different software components completely independently

**Unbundling vs. integrated systems**
If unbundling becomes the way of the future,w ill not replcae database in their current form- databases are required for maintianing state in stream processors + to serve queries

**What’s missing?**
No unbundled-database equivalent of the Unix shel
- In essence- a high-level language for composing storage and processing systems in a simple declarative way


## Designing Applications around Dataflow

TODO

## Observing Desired State

TODO



# Aiming for Correctness
TODO



# Doing the Right Thing

Let’s take a step back- data is an abstract thing, but we must treat it with respect and consider the ethics of what we build.

## Predictive Analytics
- Using data to make automated decisions about people

Increasingly important

**Bias and discrimination**
**Responsibility and accountability**
Prevent data from being used to harm people
**Feedback loops**
Cant predic then they happen, but we should employ systems thinking- what end behavior is this really enforcing?

## Privacy and Tracking

What about data collection itself?

**Surveillance** 
Consider replacing the word “data” with “surveillance” and a whole host of tech company speak turns dystopian - e.g. surveillance-driven applications

**Consent and freedom of chioce**
Recall that users havec littlte known of what data they’re feeding to our databases


**Data as assets and power**

If targeted advertising is what pays for a service, behavioral data is the core asset

**Remembering the industrial erevoutoin**
Took a long timet o establish safety regulation (think of pollution)

Bruce Schneier- “Data is the pollution problem of the information age, and protecting privacy is the environmental challenge. Almost all computers produce information. It stays around, festering. How we deal with it—how we contain it and how we dispose of it—is central to the health of our information economy. Just as we look back today at the early decades of the industrial age and wonder how our ancestors could have ignored pollution in their rush to build an industrial world, our grandchildren will look back at us during these early decades of the information age and judge us on how we addressed the challenge of data collection and misuse.”

**Legislation and self-regulation**
We need a culture shift- users are not metrics to be optimized.

# In Sum

- Solving the data integration problem with batch processing / event streams
- Data flow applications as unbundling components of a database
- How to ensure processing remains correct in presence of faults
	- Asynchronous vent processing
- Use of audits
- Ethics
