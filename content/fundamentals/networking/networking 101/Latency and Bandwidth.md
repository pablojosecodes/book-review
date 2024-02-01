---
title: Latency and Bandwidth
---
# Speed is a feeature

Empircal facts. Faster sites lead to
- Better engagement
- Better retention
- Higher conversions

2 critical components govern performance of traffic
- `latency` time from source sending a packet to destination receiving it
- `bandwidth` maximum throughput

# Components of Latency

What contributes to latency? Latency is the sum of 
- `Propagation delay`- time for message to travel from sending to reciever
	- Governed by: distance / medium of the signal’s travel
- `Transmission delay`- time to push all of packet’s bits into link
	- Governed by: data rate of transmitting link
- `Processing delay`- time to process pacet header, check for bit-level errors, determine packet’s destination
- `Queuing delay`- time incoming packet waits in queue until processed

# Speed of light and propagation Latency

TDO