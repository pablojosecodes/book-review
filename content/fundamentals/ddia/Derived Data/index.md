---
title: Derived Data
---
You should now have a solid understanding of everytig that goes into a distributed database, but what happens when there are multiple database in the application? Things get even more complex!

Let’s now discuss integrating multiple data systems into one architecture.

First, though, let’s differentiate between two kinds of systems which store / process data
- **Systems of record** / source of truth
	- Authoritative version of your data
	- Facts represented once
	- Normalized
- **Derived data system**
	- Result fo taking existing data and transforming or oprcessing it in some way
	- If you lose this data, you can recreate from original source
	- Redundant data
	- Often, denormalized


> [!TOC]
> [[Batch Processing]]
> [[Stream Processing]]
> [[Future of Data Systems]]
