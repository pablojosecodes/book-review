---
title: Data
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

# Data models + Query Languages


Data models are **extremely important**. They influence
- How software is written
- How we think about hte problem

For each layer, key question is how is it represented in terms of the next-lower layer?
1. App developer models real world with data structures + APIs
2. Data structures expressed in general purpose data model like jSON
3. Engineers who built your database software decided how to represent jSON in terms of bytes
4. Hardware engineer figured out how to represent bytes in terms of electrical currents ,etc.
May be many more intermediary layers (API built on APIs)

Clean data models allow for abstraction and embody assumptions about its usage

## Relational vs Document Model

Relational: SQL
- Data organized into relations
- Relation: unordered collection of tuples

Birth of NoSQL
- latest attempt to overthrow relational model dominance
Why NoSQL?
- Need for greater scalability
- OSS
- Specialized query operations
- More expressive + dynamic model

Object-Relational mismatch
- Impedance mistmatch: most development today done in object-oriented programming
- Data in SQL (relational) necessitates awkward translation layer

Object relational mapping (ORM) frameworks like Hibernate
- Try to reduce boilerplate for translation layer

Example of resume
- Traditional SQL model: put positions, education, contact info in separate tables
	- Foreign key reference to users table
- Later SQL: support for structured datatype- ulti-valued data in a single row
- 3rd option: Encode jobs, education, contact info as JSON document, store it on text column in database
Or, use JSON- quite appropriate

JSON has better locality than the multi-table schema- no need to make multiple queries

**Many-to-one and many-to-many relationships**

Why use ids instead of plain text strings? (ie. region_id instead of “Greater Seattle Area”)
- Consistent style / spelling
- Avoid ambiguity
- Ease of updating
- Localizaton support- dif. languages
- Easy search

Key idea: since ID has no meaning to humans, never needs to change
- Anything meaningful to humans may need to change in the fturue
- If information is duplicated, also need to change all redundant copies
	- Removing this duplication is the key idea behind **normalization** in databases

Normalizing this data requires many-to-one relationaships, which don’t fit nicely into document model
- Would have to emulate a join by making multiple queries


**Are document databases repeating history?**
Back in the day, IMS was a popular database- use da hierarchical data model (tree of records within records)
- Made many-to-many relationships difficult
- No support for joins

2 prominent solutions were proposed to solve. limitatin sof hierarchical model
- Relational model: became SQL
- Network model: eventually faded into obscurity

Network model -CODASYL model
- Generalization of hierarchical model but where a record could have multiple parents
- Allows to model many-to-one and many-to-many relationships
- Issue: access paths got complicated

Relational model
- Lay data in the open
	- No nested structures or access paths

**Relational vs document dtabases today**

Lots of differences to consider- here we’ll only discuss the differences in the data model

1. Which leads to simpler application code?
If data is doument-like: probably use document model
- Could do *shredding* for relational model, but leads to cmbersome schemas- unnecessarily complicated
- Poor support for joins may be a problem depending on the application
If application uses many-to-many relationships, document model is less appealing

2. Schema flexibility in docuemnt model
Document databses don’t enforce any schema
- Schema-on-read: sructure of data is implicit
	- Advantageous if items don’t all have same structure
- Similar to dynamic type checking

Schema changes
- All those `AlLTER` and `UPDATE` commands
- Bad reputation for being inefficient, but actually quite fast


3. Data Locality
Document stores as single contiguous string usually
Storage locality: all stored near itself
- Only advantage if need large parts of document at same time- otherwise, loading for no reason

4. Convergence of document / relational datbbases
Most relational database systems hae also supported XML
Some document databases support relaton-like joins


## Query Languages for Data

Relation model introduuced new way of querying data
- **Declarative** query language
	- Just specify the pattern of the data you want (conditions, data transformations)
	- Makes it easier to improve the underlying engine while abstracting away updates
- In contrast to previous **imperative** querying code
	- Tells compute to perform operations in a certain order

**Declarative Queries on the web**
Declarative query language also has pros when it comes to web browsers
- CSS for example `li.selected > p` e.g. declares pattern of elements

**MapReduce Querying**
`mapReduce`- programing model for processing large amount of data in bulk
Not declarative nor fully imperative query API
Based on map and reduce functions while specific filters that can be specified declaratively
Map and reduce must be pure- no side effects



## Graph-Like Data Models

Many to many relationships may be common in your data
- Starts to bcome more ntural to model your data as a graph
- e.g.
	- social graph
	- web graph
	- road / rail network

Some different ways to structure and query dat in graphs
Discussed here
- Models
	- Property graph
	- Triple-store
- Query languages
	- Cypher
	- SPARQL
	- Datalog
- Imperative graph query langauges (e.g. Gremline)
- Graph processing frameworks (e.g. Pregel)


#### Property Graph

Each vertext
- Unique ID
- Set of outgoing edges
- Set of incoming edges
- Collection of properties (key-value pairs)
Each edge
- Unique ID
- Vertex where edge starts
- Vetex where edge ends
- Label (kind of relationship)
- Collection of properties

Intuition
- 2 relational tables- vertices + edges
- Query edges table by `head_vertex` or `tail_vertex`

Important to note
1. Any vertex can connect with any vertex
2. Given any vertex, you can traverse both forward and backewards
3. Can store multiple kinds of info in a single graph while having a clean data model (different relationships, differnt labels)

#### Cypher Query Language
Declarative query laguage for property graphs (created for the Neo4j graph database)
Each vertex given a symbolic name and other parts of query can use the names to create edges between the vertices using arrow notation
```cypher
CREATE
(NAmerica:Location {name:'North America', type:'continent'}),
(USA:Location      {name:'United States', type:'country'  }),
(Idaho:Location    {name:'Idaho',         type:'state'    }),
(Lucy:Person       {name:'Lucy' }),
(Idaho) -[:WITHIN]->  (USA)  -[:WITHIN]-> (NAmerica),
(Lucy)  -[:BORN_IN]-> (Idaho)
```

When all vertices and edges are added to databse, can ask interesting questions
- e.g. “find the names of all the people who emigrated from the US to Europe”- quivalent to returning name property for vertices with:
	- BORN_IN edge to location in hte US
	- LIVING_IN edge to location in Europe
	- Query in Cypher language
```cypher
MATCH
(person) -[:BORN_IN]->  () -[:WITHIN*0..]-> (us:Location {name:'United States'}),
(person) -[:LIVES_IN]-> () -[:WITHIN*0..]-> (eu:Location {name:'Europe'}) RETURN person.name
```




cos in https://substack.com/browse/recommendations/post/140918169

# Storage and retrieval


What should a database do?
2 things
- Take in data and store it
- Give it back to you

Good to know how database handles tuff internally for picking between them
## The data structures that power your database


Most basic database as 2 hash function
```bash
##!/bin/bash

db_set() {
	echo "$1,$2" >> database
}

db_get() {
	grep "^$1," database | sed -e "s/^$1,//" | tail -n 1
}
```

`db_get` would have terrible performance for large number of records in database


Need a different, aditional data structure
- Index


**Hash Indexes**

Approaches to key-value data
1. Simple- in-memory hashmap- each key mapped to byte offset in the data file- location at which the value can be found
	1. But how to avoid running out of disk space?
	2. Break logs into segments of certian size by cosing segment file when it reaches a certain size- make subsequent writes to a new segment file
	3. Then perform compaction on these segments (throwing away duplicate keuys)
Details go into making thi practically work
- File format- Binary better than CSV
- Deleting record- append specal deletion record to data file
- Carsh recovery- if databse is restarted
- Partially written records
- Concurrency cnotrol

**STables and LSM-Trees**

Simple change- sort sequence by key
- Sorted String Table
Advantages over log segments with hash indexes
1. Merging segments is simple + efficient (is like the mergesort algorithm)
2. To find key in file- no longer need index of all the keys
	1. Can use the fact its sorted to your advantage
3. Since read requests need to scan over several key-value pairs in range anyways, can group the records into a block and compress it before writing to disk

Constructing and maintaining SStables
 - how to get data sorted by key in the first lace? Writes can occur in any order
Storage engine works as follows
- Write comes in and gets added to in-memory balanced tree data stucture (e.g. red-black ree)- this is called a memtable
- When memtable size exceeds some threshold, 

# Encoding + Evolution


Data formats change over time, how to deal with this?
- Relational databases: force 1 schema at any given time
- Non-relational: can have a mixture of old and new data formats

As data format changes, the application code also needs to
- Typical workflow = gradual, rolling upgrades to verify that it works

Ideally can maintain
- Backward compatbility: new code can read code written by old code
- Forward compatibility: old code can read code written by new code

## Data Encoding Formats

2 data representations
- Data structures: optimized for retrieval, manipulation (e.g. dictionaries, hash tables)
- Self-contained sequence of bytes
**data encoding**: data structure → byte sequence
- decoding/parsing = reverse

**language specific encoding**
Some languages have
- Pickle for Python
- Serializable for Java
- Etc.

Not good for a few reasons
1. Language-specicific
2. Decoding usualy necesitates constructing arbitrary classes- can lead to security problems
3. Generally inefficient
4. Versioning tends to ignore backwards/forward compatibility
So, only use for transient reasons

**JSON, CSV and XML**
Very common standardized encoding formats- human readable ish
Some issues
- Ambiguity with numbers- 
	- XML/CSV: no distinction b/w number and string of digits (except via schema)
	- JSON: no distinction b/w number integers and floats (and no precision)
- Large Numbers aren’t represented well
- Schemas
	- XML/JSON: optional support but get complicated
	- CSV: no schema- up to application to determine
Generally good enough for many purposes

**Binary Encoding**

Lots of binary encoding systems for JSON/XML
- Don’t change the data model
- Need to keep object field names in the data- since they don’t perscribe a schema

**Thrift and protobufs**
Both are binary encoding libraries which require a schema for any data that is encoded

To specify the schema (for example)
In Thrift
```idl
struct Person {
	1: required string userName,
	2: optional i64 favoriteNumber, 
	3: optional list<string> interests
}
```
In Protobuf
```protobuf
message Person {
        required string user_name       = 1;
        optional int64  favorite_number = 2;
        repeated string interests       = 3;
}
```

Both come with code generation tool
- take in: schema definition (like the above)
- produces: classes which implement schema in various programming languages (use to encode/decode)

How do they handle schema evolution?



**Avro**
Another binary encoding which uses schema to specify structure of data being encoded
Example schema
```idl
record Person {
	string userName;
	union { null, long } favoriteNumber = null;
	 array<string> interests;
}
```

Writer (encoding) and reader (decoding) schema 
- Don’t have to be equivalent in Avro, just compatible
- What does this mean?
	- Can have fields in different order
	- Can have less fields in reader schema
	- Enabled forward compatibility
But how does Avro know what writer schema was used for a given record?
Depends, actually
- For larger files- can include writer’s schema once at top of file
- For records with different schemas- include a version number and store schemas separately

Dynamically generated schemas
- Since Avro doesn’t include tag numbers, can esasily dynamicaly generte schema
- Example- using SQL- the column name matches on to the tag name automatically


**Merits of Schemas**

Schemas
- Valuable form of documentation
- Database of schemas allows you to check forward and backward compatibility before deploying
- Ability to generate code is nice for statically typed programming languages

## Modes of dataflow

Previous chapter
- When send data to another process with which you don’t share memory
	- Encode as a sequece of bytes 
- Backwards.+ Forward compatibility

But data flow is a pretty abstract idea
- Who encodes the data?
- Who decodes it?
Following: how data flows between different processes
#### Dataflow through databases
Process which 
- writes to database: encodes
- read from database: decodes

May be just one process accessing hte database
- “Seending message to future self'“
- Clearly has need for backward compatibility
However, can also be several accessing at same time
- May have older process writing while newer process reading
- Thus, forward compatibiity essential

Additional snag: what if process decodes and then encodes the record back
- How can it ensure it doesn’t erase newly added fields? (even though it can’t interpret it)
- Encoding formats support this, but you may need to handle at application level

**Different values at different times**
Remember that you can generally update any value at any time in a database
- May have extremely old data
- “data outlives code”
Migrating schemas is possible, but expensive
Evolving schemas make it seem like all data was encoded with the same schema (e.g. Avro)

**Archival storage**
Typically encoded using latest schema

#### Dataflow through services: REST and RPC

How to let processes communicate over a network?
- Most common: use clients and servers
	- Server: expose API (service) over network
	- Client: Connect to servers to make request to the API

How does the web work?
- Clients = web browsers, make request to web servers
	- GET: downlaod HTML/CSS, JS, images
	- POST: submit data to servers
- API: standardized set of protocols and data formats
	- HTTPS / URLs / SSL/TLS, etc.
- Since there is agreement on these standards, theoretically can access wany website with any web browser

Other types of clients (Clients are not only web browsers)
- Nativee app on movile/desktop can also make requsts to servers
- Client side app in web brwoser can use `XMLHttpRequest` to become HTTP client
- Server can be client to another service 
	- Common use case: microservice architecture

Services are kind of like databases
- Allow cliens to submit and query data
- Different because
	- Don’t have a general query language
	- Instead, expose applciation speciifc aAPI (inputs/outputs determined by business logic)

**Web services**

If HTTP is underlyuing protocl for talking to services, is a web services

Web services are not only used on web
Examples
1. Client app on user’s device
2. Making request to service of HTTP
3. Service making request to other service owned by same org, in the same datacenter (SOA - middleware)

2 popular approaches to web services (very opposed- lots of debate)
- REST- design philosophy
	- Build on HTTP principles
	- API designed according to REST = RESTful
- SOAP- XML-based protocol for network API requests
	- Aims to be independent from HTTP 
	- Most commonly used over HTTP
	- API of SOAP web services described using XML-based lanuage called WSDL

**Problem with RPCs (remote procedure calls)**

Web services are latest incarnation of
- Long sequence of technologies for making API requests over networks
- Many hyped but have real problems

Many are based on the idea of a *remote procedure calls*
RPC Model:
- Goal: make request to remote network service look like calling function in programming langauge, within the same process
	- Location transparency
- Seems convenient
- Funadamentally flawed

Network request very different from local function call!
- Predictability
	- Local function call: succeeds or fails because of you
	- Network request: unpreditable- need to account for network problems
- Outcomes
	- Local function call: result / exception / no return (due to infinite loop)
	- Network request: all of above, but can also get no return dure to timeout (no response)
- Retrying
	- Network request: May need to build idempotence into protocol- may be that only the responses are getting lost
- Execution time
	- Local function: more or less same
	- Network request: slower plus more variable for many possible reasons
- Passing arugments
	- Local function: can pass pointers
	- Network request: need to encode all dat
- Datatypes
	- Network request: May have different programming langauges for lcient + service- can be hard to translate 

In sum, no point in trying to make remote service look like local object- fundamentally different


Current state of RPC
- Not going away- lots of frameworks on it
- Currently mostly for requests between services owned by same organizatoin (within thes ame datacenter), and REST instead for public APIs

Encoding + evolution in RPC
- Forward/backwards compatibility inhereited by the encoding it uses

#### Message-passsing dataflow

So far
- REST / RPC: process send request over network to other process- expects response as quickly as possible
- Database: process encodes, anothre decodes some time in the future
Now
- Asynchronous message-passing systems

Core idea
- Client’s reqwuest (message) delivered to another process with latency
- Message sent via intermediary called *message broker* (or message queue or message-oriented middleware)
	- Why message queue better than direct RPC?
	- Allows..
	- Can act as buffer
	- Automatic redeliver message to process which crashed
	- Avoiding sender needing to nkow IP address of recipient
	- One message → several recipients
	- Decouples sender from recipient

**Message Broker**
- Don’t usually enforce a particular data model- 
	- Messages = just sequences of bytes
		- Encode however you like

**Distributed actor frameworks**

Actor model: model for concurrency in single process
- No threads- logic is encapsulated in actors
- Actor: one client or entity, maybe some local state
- Communicaiton between actors: asynchronous esnding + recieving
	- Message delibery is not maintained

Distributed actor frameworks
- Scale up actor model across multiple nodes
- Same messag-passing mechanism, no matter which node send/recipient is on
	- If different nodes- message is encoded and then decoded
- Location transparency works better in actor model than PRC
	- Actor model alread assumes message myay be lost (even within single process)
- Basically integrates message broke + actor programming model into single framework
- How to handle message encoding? Differnet distributed actor frameworks handle differently



## In sum
Encoding data structures → bytes on network or disk
Many services need to support rolling upgrades
- No dowtime with new service versions
- Need backward and forward compatibility

Data encoding formats
- Programming language specific: restricted + lack forward/backward compatibility
- Textual formats (JSON/XML/CSV): optional data schemas, somewhat vague about datypes
- Binary schema-driven formats (Thrift, protobuf, avro) compact + efficient, but must be decoded before human-readable

Modeso f dataflow
- Databases: process writes/reads
- RPC / REST APIs; Client encodes request, server decodes request, server encodes response, client decodes response
- Asynch message parsing: encoded by sender, decoded by recipient