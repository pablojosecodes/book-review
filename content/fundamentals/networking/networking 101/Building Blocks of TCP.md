---
title: Building Blocks of TCP
---

There are two protocols at the heart of the internet
- IP (internet protocol): host-to-host routing/addressing
- TCP (transmission control protocol): abstraction to give appeearance of a reliable network
	- Guaraentees: identical bytes received and in order


# 3 Way Handshake

Before client and server can exchange applicaiton data, they must agree on sequence numbers
- **SYN**: client picks random sequence number x and sends a SYN packet
- **SYN ACK**: server increments x by 1, picks random sequence y, sends response
- **ACK**: client increments x and y by 1, completes handshake by dispatching ACK packet

After handshake, application data can flow 
- Client: can send immediately after sending ACK
- Server: can after recieving ACK

# Congestion Avoidance and Control

(1984- John Nagle) documented congestion collapse
- When roundtrip time exceeds maximum retrainsmission interval- host introduces more and more cpies of the same datagram into the network
- Eventually all buffers in switching nodes are full and packets must be dropped

To address this issue, some mechanisms in TCP impemented which governed the rate at which data can be sent in both directions
- Flow control
- Congestion control
- Congestion avoidance


## Flow control
Goal: prevent sender from overhwelming receiver with data it may not be able to process

Why might reciever not be able to process the data?
- Under heavy load
- Busy
- Only willing to allocate dfixed amount of buffer space

Mechanism: each side of TCP connection advertises its receive window (`rwnd`) which communicates size of available buffer space
- On first connection: both sides initiate `rwnd` values with system default settings, then dynamically changes


## Slow-Start
Despite flow control, network congestion collapse became a real issue

Why? Flow control prevented sender from overwheleming the receiever, but didn’t have a mechanism to prevent either side from overwhelming the underlying network

Goal: find an optimal transmission speed without causing congestion

Mechanism: Initialize a new congestion window per TCP connection and set conservative initial value for new congestion window variable (`cwnd`)
- `cwnd`- not shared
- Now, maximum amount of data is the minimum of `rwnd` and `cwnd` 
- How to find the optimal window size? Start slow and grow window size as packets are acknowledged
TODO

## Congestion Avoidance
> Note: packet loss is expected- TCP is explicitly designed to use it as a feedback mechanism.

Assumption in congestion avoidance: packet loss indicative of network congestion
- vis. somewhere along path, congested link or router was forced to drop the packet, so need to adjust window to avoid overwhelming network

Improving congestion control is an active area of research!


# Bandwidth-Delay Product

Optimal sender / receiver window sizes vary based on roundtrip time + target data rate

Why?
- Maximum unacknowledged data is minimum of `rwnd` and `cwnd` window sizes

If either side exceeds maximum amount of unacknowledged data, must stop and wait for other end to ACK some of the packets before proceeding

> Bandwidth-delay product (BDP)- product of data link’s capacity and its end-to-end delay. Result: maximum amount of unacknowledged data which can be in flight at any point in time

Our goal is to minimize the gaps in data flow, which limits the throughput of the connection- so make the window sizes just big enough

How big do `rwnd` (flow control) and  `cwnd` (congestion control) window sizes need to be?
- Assume: 
	- Minimum of `cwnd` and `rwnd` window sizes is 16 KB
	- Roundtrip time is 100ms
- 16 KB / 100ms = 1.3 Mbps
	- TCP connection won’t exceed this 

How to calculate the optimal window size if we know the roundtrip time and available bandwidth?
- Assume:
	- Roundtrip time is 100ms
	- Sender: 10 Mbps available bandwidth
	- Receiever: 100 Mbps+ available bandwidth
	- No network congestion
	- Goal: saturate the 10 Mbps link
- 10Mbps / 100ms = 122.1 KB to saturate the link

> Window size negotiation and tuning is managed automatically by the network stack.

# Head-of-line Blocking
TCP isn’t the only choice- some features are unnecessary for some applications and introduce delays

Remember, TCP gives an abstraction of a reliable network running over an unreliable channel. The work done to ensure that packets arrive on order is done at the TCP layer
- Application has no visibility into TCP re-transimissions or queued packets
- All it sees is a delivery delay (known as **TCP Head of-line blocking**)

**Jitter**- subsequent unpredictable latency variation in packet arrival times

Some applications don’t need reliable / in0order delivery (TCP doesn’t support this), so may be better served by an alternate transport (e.g. UDP)

# Optimizing for TCP

Remember that TCP is an adaptive protocol designed to be fair to al network peers

To optimize TCP- tune how it senses network conditions and reacts

TCP optimization is an endless area of research

But- there are core principles and implications
- 3 way handshake introduces full roundtrip of latency
- Slow-start is applied to every new connection
- Flow + Congestion control regulate throughput of all connections
- Throughput is regulated by current congestion window size

## Tuning Server Configuration
TCP best practices keep changing, and most of these updates are only available in the latest kernels
- Basically- **just keep your servers up to date**

Also, make sure your server is configured with best practices
- Have a larger starting congestion window
	- Good for: bursty, short-lived connections
- Disable slow-start after idle
	- Good for: long-lived TCP connections that have bursts of data transfer
- Enable window scaling
	- Good for: high latency connections
- TCP Fast open
	- New- investigate if your app can use

Might need to configure other TCP settings, but those depend on platform, app, hardware

## Tuning Application Behavior 

Best practices / obvious insights
- Send fewer bits
- Move bits closer
- Reuse TCP connections
## Performance Checklist

- Upgrade server kernel
- Set `cwnd`size to 10
- Disable slow-start after idle
- Enable window scaling
- Eliminate redundant data transfers
- Compress transferred data
- Position servers close to user
- Reuse TCP connections

