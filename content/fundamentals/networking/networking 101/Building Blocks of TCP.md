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

Mechanism: Initialize a new congestion window per TCP connection and set conservative initial value for new congestion window varibale (`cwnd`)
- `cwnd`- not shared
- Now, maximum amount of data is the minimum of `rwnd` and `cwnd` 
- How to find the optimal window size? Start slow and grow dinwo size as packets are acknowledged
TODO

## Congestion Avoidance
> Note: packet loss is expected- TCP is explicitly designed to use it as a feedback mechanism.

Assumption in congestion avoidance: packet loss indicative of network congestion
- vis. somewhere along path, congested link or router was forced to drop the packet, so need to adjust window to avoid overwhelming network

Improving congestion control is an active area of research!


# Bandwidth-Delay Product

TODO