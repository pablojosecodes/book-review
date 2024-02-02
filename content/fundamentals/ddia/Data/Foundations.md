---
title: Foundations
---
# Foundations
Reliable, scalable, and maintainable applications

Typically, data-intensive appls are built from standard building blocks.
Many applications need to:
- Store data (databases)
- Remove result of expensive operation (caches)
- Allow users to search data by keyword or filter in various ways (search indexes)
- Send message to another process to be handled asynchronously (stream processing)
- Periodically crunch large amount of data (data processing)

## Thinking about data systems

The building blocks (caches, queues, etc.) wseem quite idfferent- why lump them togethr?
- Lots of overlap between categories
- No single tool can meet of data processing or storage needs
When you combine several tools, you’ve baseically createda. new special-purpose data system

When designing these things, 3 concerns are most important
- Reliability: work correctly even in face of adversity
- Scalability: should have reasonable ways of dealing with the system’s growth
- Maintainability: engineers should be able to work on it productively

## Reliability
Typical expectations
- Application performs function user expected
- Can tolerate user making mistakes
- Performance good enough for the use case under expected lload / data volume
- Prevents unauth access and abus

Our goal should be to create fault-tolerant systems
- Might want to actually increase rate of faults by triggering them deliberately

Fault vs failure
- Fault: One component deviating from its spec
- Failure: system stop providing service to user
Goal: prevent faults from causing failures

**Hardware Faults**
- e.g. hard disk crashes, faulty RAM, blackout, etc.
With > data volume and computing demands, many many machines, and thus need to tolerate the loss of entire machines

**Software Errors**
Systematic error within the system
Examples
- Software bug which caues each instance of an applcication server to crash when given a particular bad inpu
- Runaway process which uses up some shared resource
- Etc.

**Human Errors**
Inherently unreliable
Combine several approaches
- MInimize opportunities for error
- Decouple places where people make the most mistakes from where they can cause failures
- Test thorougly
- Enable quick recovery
- Clear monitoring

## Scalability
- Ability to cope with increased load
Consider key qustions e.g.
- How can we add computing resources to handle additional load?
- If system grows x way, what are our optins for coping with the growth?

**Describing Load**
Load is described with a few number- load parameters
Depends on your system’s architecture
e.g.
- Requests per second to web server
- Ratio of reads to writes
- etc.

As an example, for Twitter- the scaling challenge comes from fan-out

**Describing performance**
After describing load, can investivate what happens when load increases
2 ways of looking at it- when you increase a load parameter…
- and keep system resources unchanged, how does performance change?
- how much do you need to increase resources to keep performance unchanged?
Need to have performance numbers to think about this
- E.g. in batch processing- throughput
- Latency
- Response time percentiles

**Approaches to coping with load**
How to maintain good performance when load parameters increase?
Note: likely you’ll need to rethink your architecture on every order of magnitude load increase (maybe more often)

Scaling up vs scaling out
Distributing load across multiple machines: shared-nothing architecture
Elastic systems: can automatically add resources when detect load increase


## Maintainability
Main cost of software- ongoing maintenance

3 principles
- Operability
- Simpliciy
- Evolvability

**Operability: making life easy for Opeartions**
Operations teams need to
- Monitor health of system + be able to quickly restore
- Tracking down cause of problems
- Keep software and platforms up to date
- Keep tabs on how different systema ffect each other
- Anticipate problems
- Establish good practices/tools for deployment, etc.
- Complex maintenance tasks
- Security of system as config changes are made
- Define processes to make opeartions predictable
- Preserve knowledge
Data systems can make routine tasks easy
- Provide visibility with good monitoring
- Good support for automatiion + integration with standard tools
- Avoid dependency on individual machines
- Provide good documentation and easy-to-understand operational model
- Self-heal where appropriate

**Simplicity: managing complexity**
Large systems get complex!

Symptoms of complexity
- Large state space
- Tight coupling of modules
- Tangled dependencies
- Naming / terminology
- Et cetera

Remove accidental complexity- don’t necessarily decrease its functionality
- Accidental complexity: if not inherent in the problem that the software solves, but insteadi s from the implementation
- How? Abstraction- hide the implementation details

**Evolvability: making change easy**
System requirements will very likely change over time as you learn new facts, etc.
Agility on data system level: evolvability

## In sum

Appication has various requirements
- Functional: what it can do
- Nonfunctional: compliance, scalability, reliability, etc.
Reliability: systems work correctly even with faults
Scalability: strategies for keeping performance good
Mainainability: making life better for engineering + ops teams

