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

## Connection-State Timeouts

What’s the issue with NAT translation when it comes to UDP?

Well, NAT relies on connection state and UDP has none. TCP for example, has an established data transfer flow which lets middleman observe the state of the connection and modify routing entries as needed

So how do the translators deal with UDP? They need to keep state about each UDP flow.
They also need to figure out when to drop the translation record.

How often? No answer- depends on translator

## NAT Traversal

Another issue: inability to establish a UDP connection at all
- In presence of NAT, internal client doesn’t know its public IP
- Any packet which arrives at NAT needs destination port and entry in NAT table which can translate to internal destination host IP / port tuple

Thus- NAT is a pcacket filter- no way to automatically determine the internal route

This is not an issue for client applicaitons (TODO)

How to work around this mismatch? Some traversal techniques


#### STUN, TURN, and ICE

**STUN** (Session Traversal Utilities for NAT): protocol to let host application discover presence of NAT on the network and obtain allocated IP and port tuple for current connection
- How? Get help from third party STUN server on the public network
- Communicating with the STUN server (if IP address is known)
	- Application sends binding requst to STUN server
	- STUN replies with response with public IP + port of cliet on public network
- How does this help?
	- Application learns + can use its public IP and port tuple
	- Outbund binding request establishes NAT routing entries along the path
		- Inbound packets arriving at public IP and prot tuple can find way to host app on internal network
	- Defines mecanism for keepalive pings to keep NAT routing entries from timing out
- In practice, not sufficient to deal with all NAT configurations
	- Use TURN as a fallback

TURN
Todo


# Optimizing for UDP
In sum, UDP
- Simple, common protocol for bootstrapping new transport protocols
- Key features are its nonservices
- You’ll need to build many features from scratch if you use

Make sure you research / read best practices if you use UDP. Applications.. (following is short sampe of rec.s)
- Must
	- Enable IPv6 checksum
- Should
	- Control rate of transmission
	- Perform congestion control over all traffic
	- Use bandwidth similar to TCP
	- Back off retransmission counters following loss
	- Not send datagrams which exceed path MTU
	- Handle datagram loss / duplication / reordering
	- Be robust to delivery delays up to.2 minutes
	- Enable IP4 UDP checksum
- May
	- use keepalives when neded




