---
title: The Trouble with Distributed Systems
---
We’ve discussed a lot of ways that systems can go wrong, but you should really assume that **anything which can go wrong will go wrong**.

# Faults and partial Failures

On a single computer: software either works or it doesn’t. But really, computers hide the fuzzy physical reality on which they’re implemented.

In distributed systems, however, failures are nondeterministic and you can get partial failures.


## Cloud Computing and Supercomputing
Building large scale computing systems- you have 3 philosophies
- **High performance computing**: supercomputers with 1000s of CPUs
- **Cloud computing**: multi-tenant datacenters, elastic resource allocation
- **Traditional enterprise datacenters**: somewhere between

Internet services and cloud computing are very different than supercomputers
- Need low latency and high availablility
- Hardware aren’t as specialized
- Supercomputers can use specialized setups
- Can kill and request replacement for node 
- Communication may be very slow between nodes

# Unreliable Networks
Remember- distributed systems we’re talking about are **shared-nothing systems**- their communication is mediated by the network.

Most of these networks are **asynchronous packet networks**: no guarantee of packet arrival time or whether it will in fact arrive. Many things can go wrong
1. Lost request
2. Request in queue
3. Remote node failure
4. Temporary node pause
5. Lost response
6. Delayed Respose

How to handle this? Typically by using a timeout

#### Network Faults in Practice
These problems are surprisingly common even in data centeres, according to real world studies

This makes being able to response to these faults non-negotiable

## Detecting faults
When can you actually get specific feedback saying something isn’t working?
- Trying to reach machine on which node is running, but process crashed, the OS  will refuse TCP conection by sending a RST or FIN packet in reply
- If have management interface of network switches- can query to find link failure

Typically- use timeouts and declare node dead 
## Timeouts and unbounded delays
How long to make timeouts when declaring nodes dead?

Sadly, most real world systems have **unbounded delays**- o limit on how long packet may take to arrive

#### Network congestion and queueing
What causes variabillity of pakcet delays on computer networks? Typically, due to **queueing**. Some ways this can occur
- Scenario: Several different nodes sending packet to same destination
	- Impact: Network congestion- Network switch queeues and feeds to destination network link 1-by-1
- Scenario: Packet reaches destination machine but all CPU cores busy
	- Impact: Incoming request queued by OS
-  Scenario: In virtualized environments, running OS paused while another VM uses CPU core
	- Impact: Incoming data queued by VM monitor
-  Scenario: TCP performs **flow control**- node limits rate of sending to avoid overloading network link / receiving nodes
	- Impact: queueing at the sender itself

## Synchronous vs Asynchronous Networks
Why can’t we make the network more reliable?

Compare to fixe-dline telephone network- its extremely reliable. Why can’t we have the same for computer networks?
- How does telephone network work? 
- Calls establish a **circuit**- guaranteed amount of bandwidth- until call ends
- This is a **synchronous** network- the bits of space fr or call are all reserved- no queueing
	- **Bounded delay**- the fact that there is a max end-to-end latency becuase network is fixed

#### Can’t we just make network delays predictable?
TODO

# Unreliable Clocks

Why do we care about clocks in our applications?
- Measuring duration
	- Timeouts
	- Percentiles
	- Queries per second
	- +++
- Describing points in time
	- When to send?
	- When does this expire?
	- +++

In distributed system, time gets complicated as communication isn’t instantaneous. Synchronization is usually done with the Network Time Protocol (NTP).

## Monotonic vs. time-of-day cocks
Modern computers have at least 2 different clock types.

#### **Time-of-day clocks**
**Purpose**: measure the current date and time

**Functionality**: usually synchronized with NTP

#### **Monotonic clock**
**Purpose**: measure a duration, such as timeout or response time

**Functionality**: the actual value of the clock is meaningless- what matters is difference between two values at different points in time. 
- NTP can adjust frequency of monotonic clock if it detects the computers local quartz moving faster or slower than the NTP server. 

## Clock Synchronization and Accuracy
Tie of day clocks need to be set accordint o time source. However, not as accurate as we’d like
- Quartz clock in computer isn’t very accurate- it drifts (assume 17s drift if resyncronized once a day)
- If node is accidentally firewalled off from NTP servers, misconfiguration may go unnoticed
- NTP synchronization is only as good as network delay
- Leap seconds can mess up large systems
- Can’t trust device’s hardwareclock at all if on device you can’t fully control (user may have modified)

## Relying on synchronized clocks
Incorrect clocks easily go unnoticed- most things will seem to work fine

#### Timestamps for ordering events
Tempting to rely on clocks to order events across multiple nodes

Important to rememeber that “recent” depends on a local time-of-day- clock which may be incorrect. 

### Clock readings have a confidence interval
Don’t think of a clock reasing as a point in time- think of it as a range of times within a confidence interval (ie. 95% confidence time is between 10.3-10.5 seconds)

How to calculate this uncertainty bound? Well, atomic clock will have an expected error range form the manufacturer. But most systems won’t expose this uncertiainty (quartz drift plus NTP server uncertainty plus round trip time to server)

#### Synchronized clocks for global snapshots
Remember snapshot isolation: allowing read-only transactions to see database in consistent state

Can we use timestamps from synchronized time-of-day clocks as transaction IDs? Then we could directly use them to determine when transactions took place

perhaps- if you incorporate confidence intervals- for example, Spanner waits for length of confidence interval before committing read-write transaction

## Process Pauses

Another case of dangerous clock usage- datbase with single leader per partition.

How does node know it’s still leader and can accept writes?
- One option: obtain a lease from other nodes which lasts a specific duration. 
- BUT 
	- This relies on synchronized clocks
	- Thread might be paused (this can happen for a very long time, where thread can be preempted without it noticing)

#### Response time guarantees
TODO
#### Limiting the impact of Garbage Collection
TODO

# Knowledge, Truth, and Lies
We’ve talked about how distributed systems have no shared memory, operate via an unreliable network with variabel delays, may suffer from partial failures, ureliable clocks, and processing pauses.

A node cannot know anything for sure- how can we establish a systematic way to ascertain truth and so on about our data system?

We can state the assumption we make about hte behavior (the system model) and desing system in a way that meets those assumption

Lets’s talk about what assumption we can make by exploring defintions of knowledge and truth in distributed systems.

## Truth is Defined by the Majority
A node cannot trust its own judgement and a distributed system cant exclusively rely on a single node. Any node can fail at any time.

Many distributed algorithms rely on a **quorum**- voting among the nodes (including voting nodes dead)

#### Leader and the lock
We often need exactly one of something
- One leader for partition
- One transaction can hold lock
- One user for a username

This requires care- node may think its leader, but if quorum doesn’t agree- it is not.


#### Fencing tokens
**Fencing**: a technique to ensure that node which is under false belief of being “chosen on” doesn’t disrupt rest of system (e.g. when using lock / lease to protect access to a resource)
- Fencing token given to owner of lock (with number that increments for each lock given)
- Then- storage server can check writes against its current lock number
- Note that this requires resource to check the tokens

## Byzantine Faults

Fencing tokens can detect + block node which is accidentally acting in error- but can’t block node which deliberatey uses a fake fencing token

**Byzantine faut**: when there is a risk that the nodes may lie (send faulty/corrupted responses)

**Byzantine fault-tolerant system**: f continues to correctly operate even if malicious attackers interfering or nodes are malfunctioning.
- When would we need this?
	- Aerospace: corruption by radiation
	- Multiple organiztions: may have participants try to cheat

Usually, though, you can assume no Byzanteine faults


#### Weak forms of lying

Even though w assume nodse are honest, can be worth adding mechanisms to guard against weak lying
- Weak lying: invalid messages due to hardware issues, bugs, misconfigs

## System model and reality
Algorithms can’t depend too much on the details of the configuration on which they are run

Thus, we need to somehow formalize the faults we expect-

Timing assumptions- 3 common system models
- Synchronous model
	- Assume: 
		- Bounded network delay
		- Bounded process pauses
		- Bounded clock error
	- Not realistic
- Partially synchronous model
	- Assumes
		- Synchronous system most of the time
		- Sometimes exceeds bounds for network delay, process pauses, clock drift
	- Realistic
- Asynchronous model
	- NO timing assumption- no clock at all
	- Only some algorithms can be designed for this model

Model systems for nodes (we need to also consider node failures)
- Crash-stop Model
	- Assume
		- Node can only fail by crashing (ie. node that stops responding is gone forever)
- Crash-recovery model
	- Assume
		- Nodes may crash
		- Nodes may start responding after unknown time
		- Nodes have stable storage (i.e. SSD) which is preseved across crashes
		- In-memory state is lost
- Byzantine faults
	- Assume
		- Nodes may do anything

Most useful model: **Partially snchronous model with crash-recovery faults**

#### Correctness of an algorithm
What  does it mean for an algorihtm to be correct?

Well, we want specific properties for a given task. If it satisfies those properties, it is correct

As an example, for an algorithm that generates fencing tokens for a lost, we may want these properties:
- Uniquess
- Monotonic sequence
- Availbility

We can see that an algorithm is **correct** if it satisfies its properties in all situations that we assume may occur in that system model.


#### Safety and liveness
There are 2 kinds of properties that are useful to distinguish between: safety and liveness

- Safety properties: “nothing bad happens”
	- Formally: If safety property is violeated, we can point to specific point at which it was broken
	- Cannot be undone
- Liveness properties: “something good eventually happens”
	- Formally: may not hold at some point in time, but always ope it’ll be satisfied in teh future

In distributed algorithms, it’s common to require that safety properties always hold

# In Sum

We talked about some of the partial failures that occur in distributed systems
- Packet or reply may be arbtrarily delayed or lost
- Node’s clock can be out of sync or jump forward/backward in time
- Process can pause at any point for a substantial amount of time (and spring to life)

To tolerate faults
- Detect (hard)
- Deal with them (also hard)

Bounding delays and giving response guarantees in networks

Supercomputers

Pretty bleak outlook in this section- see future sections for more optimism!