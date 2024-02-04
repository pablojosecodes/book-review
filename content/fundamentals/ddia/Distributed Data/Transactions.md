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

No! We can use the magic of **transactions** to simplify these issues

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
1. Dirty reads: seeing data which hasn’t been committed
2. Dirty writes: overwrite data which hasn’t been committed
#### Dirty Reads
Goal: make writes visible to others only when transaction commits
- In multi object updates- prevents one aspect seeming updated while another isn’t (e.g. profile photo updates, name doesn’t)
- In case of abortion- prevents you form reading data which is never committed!
#### Dirty Writes
Goal: prevent write from overwriting uncommitted value
- In multi object updates- prevents conflicting values in table (e.g. two bids submitted at same time)
- Does NOT prevent race conditions

#### Implementing Read Committed

Preventing Dirty Writes: Use row level locks
- Transaction must acquire a lock on object until transaction is committed

Preventing dirty reads:
- One option: use similar lock- require transaction to lock during reading
	- However- this could cause long write transaction to stall a whole bunch of read-only transactions
- More popular option: database remembers old committed value while transaction holds write lock- so reads read the old value if object is locked

## Snapshot Isolation and Repeatable Read

What could still go wrong with read-committed isolation?
- Nonrepeatable read / read skew: when transactions modify data between reads in a multi object transaction
- When is this temporal inconsistency NOT acceptable?
	- Backups
	- Analytic queries / integrity checks

How do we solve this?  **Snapshot isolation** is the most common solution
- Core idea: each transaction reads from a consistent snapshot of the database

#### Implementing Snapshot Isolation

Use write locks to prevent dirty writes

**Multi version concurrency control**: store several different committed version of object, since various transactions may need state from different points in time.

How does this work?
- Each row has
	- `created_by` field: contains ID of transaction which inserted row
	- `deleted_by` field: if marked for deletion, set to ID of transaction which requested
- When is certain no transaction will want to access the deleted data, garbage collection removes rows marked for deletion

#### Visibility rules for observing consistent snapshot

How do you determine what objects are (in)visible? Based on Transaction IDs

This is essential for presenting a consistent snapshot of the database. Here’s how it works
1. Beginning at the start of transaction- database makes list of other transactions which are in progress at that time (these are ignored)
2. All writes made by aborted transactions are ignored
3. Writes made by transactions with later transaction ID are igored
4. All other writes are visible

So, an object is visible it
- At time of reader’s transaction, transaction which created object already committed
- Object is not marked for deletion or transaction which marked it had not committed when reader’s transaction started

#### Indexes and snapshot isolation
What about indexes in a multi-version database?
- One option: index points to all versions of object
	- Use index query to filter any object versions not visible to current transaction
- Other option: append only / copy on write variant- doesn’t overwrite paes of tree when updated, but instead creates new copy of each modified page 
	- Each write transaction = new B-Tree rot
	- No need to filter based on transaction IDs- subsequent wites can’t moidify existing B-tree: only create new tree roots

#### Repeatable Read and Naming Confusion
Snapshot isolation has many names
- Serializable: Oracle
- Repeatable read: PostgreSQL and MySQL

## Preventing Lost Updates
We’ve discussed guaranteeing the quality of read-only transactions, but what about two concurrent transaction writes?

This is called the **lost update** prolem
- **Specific definition**: 
	- Application reads value from database, writes back the modified value
	- But another transaction concurrently read before the writing
	- That first update would be lost
- Examples: multiple users/transactions-
	- Increment counter
	- Local change to complex value
	- Editing page

Here are some solutions!

#### Atomic write operations
**The idea**: make writes irreducible

**Implementation**: take exclusive lock on object (no other transactions can read) when its read until update is applied.
- Another option: execute atomic operations on a single thread


Here's how you can complete your notes based on the provided context:

#### Atomic write operations
**The idea**: Make writes irreducible

**Implementation**: 
- Take exclusive lock on the object when it's read so no other transaction can read it until the update is applied. This is sometimes known as cursor stability.
- Another option: Execute atomic operations on a single thread.

#### Explicit Locking
**The idea**: Application explicitly locks objects which are going to be updated.

**Implementation**:
- Use specific database commands like `SELECT ... FOR UPDATE` to lock the rows that are going to be updated.
- `FOR UPDATE` tells db to lock all rows returend by query

#### Automatically detecting lost updates
**The idea**: Parallel read-modify-write cycles and aborting + retrying when lost updates are detected.

**Implementation**:
-  Use snapshot isolation to make this efficient

#### Compare-and-set
**The idea**: Only allow updates if value hasn’t changed since you last red it.

**Implementation**:
- Note: Ensure the update is retried if not allowed to update

#### Conflict Resolution and Replication
**The idea**: In replicated databases, this can get quite a bit more complex. Locks and compare-and-set assume a single up-to-date copy. Instead, typical approach is to allow for siblings and use applicatoin code / data structures to merge versions after hte fact

**Implementation**:
- Allow concurrent writes to create conflicting versions of a value (siblings) and then resolve and merge these versions after the fact.

## Write Skew and Phantoms
We saw 2 race condition types which occur when transactoins currently write to same objects: *dirty writes* and *lost updates*.

But, there are subtler examples of conflicts

Imagine: must have at least one doctor on call- 2 on duty doctors check out at exact same time

#### Characterizing Write Skew
Ths is called **write skew**- isn’t a dirty write nor a lost update (two transactions are updating tow different objects)

Generalization of lost update problem

How to fix?
- Atomic single object ops don’t help
- Automatic lost update detection doesn’t help either
- Some databases allow constraints
- If you can’t use a serializable isolation level, the second-best option in this case is probably to explicitly lock the rows that the transaction depends on


#### More examples of write skew
- Meeting room booking
	- Trying to avoid double booking
- Multiplayer game
	- Avoid multiple players moving figure at same time
- Username claiming
- Preventing double spending

#### Phantoms causing write skew
All of these examples follow a similar pattern:
1. SELECT query checks whether some requirement is satisfied by searching for rows that match some search condition
2. Application code decides how to continue
3. If the application decides to go ahead, it makes a write

#### Materializing conflicts

If the problem of phantoms is that there is no object to which we can attach the locks, perhaps we can artificially introduce a lock object into the database?
For example, in the meeting room booking case you could imagine creating a table of time slots and rooms. Each row in this table corresponds to a particular room for a particular time period (say, 15 minutes). You create rows for all possible combina‐ tions of rooms and time periods ahead of time, e.g. for the next six months.
Now a transaction that wants to create a booking can lock (SELECT FOR UPDATE) the rows in the table that correspond to the desired room and time period. After it has acquired the locks, it can check for overlapping bookings and insert a new booking as before. Note that the additional table isn’t used to store information about the book‐ ing—it’s purely a collection of locks which is used to prevent bookings on the same room and time range from being modified concurrently.
This approach is called materializing conflicts, because it takes a phantom and turns it into a lock conflict on a concrete set of rows that exist in the database [11]. Unfortu‐ nately, it can be hard and error-prone to figure out how to materialize conflicts, and it’s ugly to let a concurrency control mechanism leak into the application data model. For those reasons, materializing conflicts should be considered a last resort if no alternative is possible. A serializable isolation level is much preferable in most cases.

# Serializability
