---
title: Storage and Retrieval
---
# Storage and Retrieval


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

