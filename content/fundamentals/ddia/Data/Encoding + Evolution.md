---
title: Encoding + Evolution
---
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