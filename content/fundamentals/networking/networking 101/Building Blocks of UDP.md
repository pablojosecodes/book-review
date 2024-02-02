---
title: Building Blocks of UDP
---
What is UDP? It stands for User Datagram Protocol, commonly referred to as a “null protocol”, and is a simple, stateless protocol.

**Some history**: introduced during same era as when TCP and IP were split into separate RFCs

What does UDP do?
- Let’s figure it out
- Some common uses
	- WebRTC
	- DNS


# Null Protocol Services

First, remember the IP, which is a layer below both TCP and UDP
- Main goal: deliver datagram from source to destination host based on addresses
- Does not: guarantee anything about message delivery/notifs of failure
	- This is the responsibility of layer above it!

UDP structure
- Adds 4 additional fields
	- Source port (optional)
	- Destination port
	- Length of packet
	- Checksum (optional)
- On delivery by IP, host can unwrap UDP packet, identify target application (by the destination port) and deliver the message

The most important parts of UDP are what it **doesn’t** do. Here are its non-services
- No **guarantee of message delivery** 
	- No acknowledgements, retransmissions, timeouts
- No **guarantee of order delivery**
	- No packet sequence numbers, reordering, head-of-line blocking
- No **connection state tracking**
	- No connection establishment, teardown state machines
- No **congestion control**
	- No build-in client / network feedback mechanisms


# UDP and Network Address Translators
IP NAD (Network address trnaslator) specification
- Temporary solution to the IPv4 address depletion problem
- How?
	- Introduce NAT devices at edge of network,
	- Responisible for maintaining a mapping of local IP and port tupels to one or more globallyunique IP an port tuples
- BUT
	- Become permanent and ubiquitous
	- Now we need to deal with them

TODO