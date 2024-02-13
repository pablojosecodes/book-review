---
title: Stream Processing
---
We iscussed batch processing and saw how the output is a form of *derived data*

But, we made a big assumption- that the input is bounded

In reality- data is unbounded very often- can gradually arrive

Stream processing
- Stream: data incrementally made available
- Event stream: Unbounded incrementally processed counterpart to the batch data from the last chapter

# Transmitting Event Streams
Event: basically record- small, immutable object
- e.g. user action
- Could be stored in text / JSON / binary
- Generated once by a producer, then potentially processed by multiple consumers
- Grouped into topic or stream

In principle, could connect consumers and producers iwth a file / database, but this polling can get expensive 
- Notifications are more effective

## Message System
This is a common approach for notifying ocnsumers about new events
- Producer sends message contianing event
- Event pushed to consumers





# Databases and Streams

We’ve seen how message brokers have taken ideas from databaes and applie them to messaging, but what about the reverse
- Take ideas frommessage / streams and apply them to database

Remember event: something which happened at some point in time (including write to a database)
- Fundamental link between databases and streams

Replication log- stream of database write events!

## Keeping systems in sync

No single system can work- usually need several, all of which need their own data and so on
- Need to keep them all in sync

If full database dumps re two slow, dual writes are occasionally used- app explicitly writes to each of the systems when data changes
- Isses
	- Race condition
- Ideal situation would instead be having a single leader, maje other system a follower


TODO

# Processing Streams
TODO

# In Sum
