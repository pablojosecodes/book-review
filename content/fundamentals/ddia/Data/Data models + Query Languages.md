---
title: Data models + Query Languages
---
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

