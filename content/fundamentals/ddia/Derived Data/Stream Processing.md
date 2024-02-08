---
title: Stream Processing
---
We iscussed batch processing and saw how the output is a form of *derived data*

But, we made a big assumption- that hte input is bounded

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
TODO

# Processing Streams
TODO

# In Sum
