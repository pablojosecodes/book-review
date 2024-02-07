---
title: Batch Processing
---
So far we’ve talked about requests/queries and responses/results (e.g. HTTP-REST-based APIs), but this isn’t the only way to build data systems

3 types of systems
- Services (online)
	- Service- waits for request and then handles with response as quickly as possible
	- Metrics
		- Reponse time (primary)
		- Availability (also important)
- Batch Processing Systems (offline)
	- Take in large amount of input data, runs job, and produces output
	- Metrics
		- Throughput (primary)
- Stream processing Systems (near real-time)
	- Somewhere betwee online / batch processing
	- Operates on events shortly after they happen


Why do we care about batch processing? It’s extremely important for scalable apppicaitions (e.g. MapReduce)

# Batch Processing with Unix tools

Take a simple example: web server with line in log file for each request (with info for who user is, what they requested, web browser,  etc.)


## Simple Log Analyis
Usually you’ll use existing tools, but you can build your own Unix tools

```unix
cat /var/log/nginx/access.log | 1
      awk '{print $7}' | 2
      sort             | 3
      uniq -c          | 4
      sort -r -n       | 5
      head -n 5        | 6
```

1. Read log file
2. Split line in whitespace and print 7th word (requested URL) 
3. Sort requested URLs
4. Filter out repeating lines
5. Sort by number of times URl was requested
6. Output 5 first lines

**chain of command vs. custom program**
Alternative would be to write your own quick command (e.g. Ruby or Python)

Picking between the two methods is a matter of taste, but there are some considerations

- Sorting vs in memory aggregation
	- Hash table approach: if small enough data, can easily fit all URLs
	- Sorting approach: makes more efficient use of memory

## Unix Philosophy
Unix Philosophy: set of design principles popular among devs/users of Unix

1. Make each program to done thing well
2. Expect output of every program become iput to another
3. Design to be tried early on

Basicaly: automation, rapid prototyping, etc.

**Uniform interface**
If output = input somewhere else, need a compatible interface

In Unix- interface is a file (really, file descriptor)
- With a uniform interface of ASCII text

**Separationfo logic and wiring**
Another characteristic feature of Unix tools: use of `stdin` and `stdout`

You can use pipes to run these all over

To participate in data processing pipelines
- Just needs to read from `stdin` and write to `stdout`

**Transparency and Experimentation**
It’s also quite easy to see what’s going on in Unix tools
- Input files: treated as immutable
	- Run commands as often as you want
- Can end pipeline at any point- pipe the output into `less`
	- Debug very easily
- Can write output to file and use file as input later
	- Split pipeline into chunks


# MapReduce and Distributed Fileystems

MapReduce: programming framework for writing code to process large datasets in a distributed filesystem
- Is like Unix tools, but distributed
- Blunt but effective tool
- MapReduce job $\approx$ single Unix process
	- 1+ input → 1+ output
- Instead of `stdin` and `stdout`, distributed file system (e.g. Hadoop’s HDFS)


HDFS (Hadoop File System):
- Shared nothing principle
- Daemon process on each machine exposing network service which allow snodes to access files stored on the machine
- `NameNode` server racks which file blocks stored on which machine
	- Conceptually: HDFS creates big filesystem which can use space of all the machines running the daemon
- Dat is replicated on machines

### MapReduce Job Execution

In order to create a MapReduce job, you need two callback function
- Mapper: extract key + value from input record
	- Called once for every input record
	- May generate any number of key-value pairs
- Reducer: 
	- Takes key-value pairs, collects those with same key, reducer iterates over collection of values


**Distributed Execution of MapReduce**
TODO

**MapReduce Workflows**
TODO

## Reduce-Side Joins and Grouping
How exactly are joins implemented?

In many datasets, foreign keys (relational) / document references (document model) / edge (document model) are common

Joins are needed when you want to access reocrds on both sides of the association

MapReduce performs **full table scan**
- When MapReduce given a set of files as input- reads the entire content

**Example: analysis of user activity events**
TODO

**Sort-merge joins**
TODO

**Bringing related data together in the same place**
TODO

**GROUP BY**
TODO

**Handline Skew**
TODO


## Map-Side Joins
TODO
## Output of batch workflows
TODO

## Comparing Hadoop to Distributed databases

Hadoop is somewhat like a distributed version of Unix, where HDFS is the filesystem
- Very similar to maively parallel processing databases

The biggest difference is that
- MPP databases: focus on parallel execution of analytic SQL queries on a cluster of machines,
- Combination of MapReduce and distributed filesystem: more like a general-purpose OS which can run arbitrary programs

**Diversity of storage**
- Hadoop: opened up the possibility of indiscriminately dumping data into HFS and worrying about further processing later.
- MPP databases: needed careful upfront modeling of data nd query patterns

This is why Hadoop is often used for ETL processes
- Data from transaction processing = dupmed into distributed filesystem
- Then- MapReduce jobs written toc lean up the data + tansfomrm into relational form, import into MPP data warehouse


**Diversity of processing models**
 - MPP: monolthic tightly integrated pieces of software
 - MapReduce: flexibility to run your own code over arge datasets

**Designing for Frequent Faults**

On failure of task
- MPP: abort ientire query
- MapReduce: can tolerate failure of map / reduce task by retyring work at granularity of individual task
	- Also- eager to write data to disk
	- More appropriate for larger jobs


# Beyond MapReduce

MapReduce is just one among many programming modes for disributed systems

We talked so much about MapReduce because it is a useful learning tool- clear abstraction on top of a distributed filesystem

Although- MapReduce is quite difficult to use directily- you would need to e.g. implement any join agorithms from scratch

Now, let’s look at some alternatives for batch processing

## Materialization of Intermediate state
TODO

## Graphs and Iterative Processing
TODO

## High-level API and Languages

Attention has turned to 
- Improving programming model
- Improving efficineccy of processing
- Etc.

APIs + higher level languages popular isince MapReduce programming is laborious

Dataflow PAIs
- Relational sytyle building lbocks to express computation
- Allow interactive use
- Both improve human interface + job execution efficiency

**Move toward declarative query languages**

Specifying join as relational operators vs. spelling out code
- Framework can anlyze properties and decide which join algorithm to use
- Can make big difference
TODO



# Summary

Unix

problems which distributed batch processing frameowkrs need to solve
- Partitioning
- Fault Tolerance

Join algorithms for MapReduce
- Sort-merge joins
- Broadcast hash joins
- Partiitoned hash joins

Resitricted programming model:
- Callback functions assumed to be stateless
- Allows for retries safely

Batch processing job
- Input data is bounded
- Reads some input data and roces output data without modifying Input data