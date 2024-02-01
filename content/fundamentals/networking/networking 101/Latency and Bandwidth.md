---
title: Latency and Bandwidth
---
# Speed is a feature

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

# Speed of light and Propagation Latency

Light is fast. Very fast. 

But our packets travel through a medium like wire or fiber optic cable, which slows it down.

The **refractive index** of a material detemines the speed of light in that material. Larger refractive index → slow light travel.

Rules of thumb: 
- Seped of light in fiber- 200M meters / second. This is a refractive index of ~1.5
- 100ms delay- “laggy”
- 300ms delay- “sluggish”
- 1000ms delay- context switch


# last mile latency

Infamous last-mile problem: the real latency is often introduced in the “last mile”, not traversing the oceans. Often takes as much time as traveling across a continent would.

Test it out by doing `traceroute website.com`

# Bandwidth at the Network Edge
Fiber links can move hundreds of terabits per second, but edges of network have much much less capacity and varies based on deployed tech

Examples
- Dial up
- DSL
- Cable
- Fiber-to-the-home

User’s bandwidth- lowest capcity link between client and destination server

## Delivering higher bandwidth + lower latencies

How to increase
- Bandwidth: add more fiber links and make better tech
- Latency: we’re almost to the speed of light already…

So, we need to architect + optimize protocols / networking code with awareness of limiitations of available bandwidth and the speed of light. 
- Hide latency through caching
- Reduce round trips
- Move data near clients
-