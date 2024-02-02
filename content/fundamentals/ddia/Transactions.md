---
title: Transactions
---
There are a lot of potential issues in data systems
- Hardware/software may fail
- App may crash
- Network may fail
- Several clients may write to db at same time
- Client may read partially updated data
- Race conditions between clients

How can we mitigate all these faults? Should we just give up?

No! We can use the magic of **transactions** to simply these issues

A transaction is a mechanism for applicaitons to group relevant reads and writes into one bundle. This turns all the actions’ success into something which can be measured in a binary way. Either the transaction succeeded or it failed.

# Slippery Concept of a Transaction
Where do we find transactions in modern data systems? Virtually all relational databases (and some non-relational) support transactions.

However, it is important to understand the tradeoffs and limitations of transactions.

## ACID
Transactions are often characterized by the concept of ACID, but ACID isn’t a very specific 1-to-1 term. Let’s define each component of ACID and in turn, solidify our understanding of transactions.

#### Atomicity
**Folk definition**: Atomic entities are generally understood to be entities which cannot be further broken down (e.g. threads in multi threaded programming)

**In the context of ACID**, atomicity:
- DOES NOT 
	- Describe what happens when several processes concurrently attempt to access data
- DOES
	- Describe what happens if clients want to make several writes, but fault occurs after some of the writes have been processeed.

Atomicity in transactions ensures that a transaction must be fully completed or else is aborted.

#### Consistency
**Folk definition**: Consistency describes a whole bunch of things! It has different definitions depending on whether you’re referring to replica consistency, consistent hashing, CAP theorem, or ACID

**In the context of ACID**, consistency is all about defining certain things which will always be true about your data. 

>Note: Since your database canot stop ou from writing bad data that violates your invariates, this must be handled in the **application** layer.

### Isolation
**Problem solved**: imagine you have two clients incrementing a counter simultaneously (vis. client reading value, incrementing, passing back). May get just one increment!

**In the context of ACID**, isolation means that concurrently executing transacions are isolated from eaech other.

#### Durability
**In the context of ACID**, durability promises that data that a transaction has successfully sent won’t be forgotten 

How is this guaranteed?
- Single node database: typically write data to SSD and a write-ahead log. 
- Can also imagine replicability providing a level of durability

> Note: perfect durability cannot exist

## Single and Multi-Object Opeartions

Remember, atomicity and isolation describe what to do if client makes several writes within the same transaction
- Atomicity: discard operations if transaction fails
- Isolation: concurrent transactions shouldn’t interfere

**Multi object transactions** require you to know which operations belong to which transaction, in order to enable A and I

How? 
- In relational databases- base it on TCP connection to db server. Anything between `BEGIN TRANSACTION` and `COMMIT` statement is considred to be part of the same transaction. 
- In nonrelational databases, no way to group operations together

### Single object writes
Atomicity and isolation apply to single objects too. Can you think of how?

As an example: let’s say you’re writing a gigantic document to a database. Your data system should be able to handle
- Connection failing halfway through
- Power failure
- Client reading document while rwrite is in progress

Let’s avoid this! And have atomicity and isolation on the level of a single object on a single node.

> Note: single-object operations are not technically transactions but are sometimes colloquially referred to as *light-weight transactions*

#### Why do we need multi-object transactions?

Why don’t we just use single-object operations and call it a day? Why bother with the additional complexity?

Well, here are some key cases where you really do need multi-object transactions
- Relational data: to use foreign keys
- Document data model: updating denormalized information
- Secondary indexes: basically in order for these to exist, you need multi-object transactions

#### How to handle errors and aborts?
Here are some things to keep in mind
- Datastores with leaderless replication tend to just give their “best effort” and notify of errors- they pass on the responsibility of errors to the application
- What should you do in cases like these? You might be tempted to just retry aborted transactions, but you should be aware of some failure states
	- If network failed: might be writing transaction twice
		- Safeguard: application level deduplication
	- If error is due to overload: just makes the problem worse
		- Safeguard: backoff, limited retries
	- If error permanent: no point


# Weak Isolation Levels

What is necessary for concurrency issues to be a concern? Well, when a transaction reads or writes data concurrently with another transaction.

Fully serializable isolation is the strongest form of isolation, but many data systems don’t want to pay the performance cost. Thus, many use **weaker isolution levels**.

Thus, while systems try to hide concurrency issues from app developers via transaction isolution, it’s important to have a good understanding of them and some mechanisms for prevention.

## Read Committed

This is the most basic level of transaction isolation. It promises two things won’t occur:
1. Dirty reads: seeing data which hasn’t been commited
2. Dirty writes: overwrite data which hasn’t been commited
#### Dirty Reads
Goal: make writes visible to others only when transaction commits
- In multi object updates- prevents one aspect seeming updated while another isn’t (e.g. profile photo updates, name doesn’t)
- In case of abortion- prevents you form reading data which is never commited!
#### Dirty Writes
Goal: prevent write from overwriting uncommited value
- In multi object updates- prevents conflicting values in table (e.g. two bids submitted at same time)
- Does NOT prevent race conditions

#### Implementing Read Commited
TODO