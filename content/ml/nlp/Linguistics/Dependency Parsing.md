---
title: Dependency Parsing
---

Base info

Constituency = phrase structure grammar = context-free grammar (throwback)

**Phrase structure**- organize words into nested constituents
- Words → phrases → bigger phrases

Example definiions of phrases
- Define a noun phrase  NP = det (adj)$^*$ N (PP)$^*$
- Same for PP → P NP

**Contrast with dependency structure**
- what modifies/depends on what?
- Represent with arrows

**Why do we need this anyhow?** 
- Human language is complex, represent abstract concepts + complicated things with combninations of words
- Ambiguity 
	- Prepositional phrase attachment ambiguity
		- Which can multiply
	- Coordination scope ambiguity (mod of what?) 
		- E.g. first hand job experience
	- Verb phrase attachment ambiguity
		- e.g. mutilated body on Rio Beach to be used for Olympics beach volleybal
- Dependency phths helps extract semantic interpretation (e.g. in a drug context)






**Dependency Structure**

Parse trees in NLP
- Use to analye syntactic structure of sentences

Dependencies: the binary asymmetric relations (Often arranged in a tree tructure)
- Arrow from head to dependent, usually typed with name of grammatical relations
- Connects head with dependent
- Usualy add a fake root so every word depends on one node

*Treebanks*: 
- Annotated data of dependencies in sentences
- Why?
	- Reuse labor (can build parsers on it)
	- Broad coverage
	- useful frequencies + distributional information
	- Evaluation method

**Dependency conditioning** 

Good sources of info for dependency parsing
- Bilexical affinities
- Dependency ditance
- Intervening material (lots of nouns?)
- Valency of heads (how many dependents are usual for a head and which side?)


**Dependency parsing**

Goal: mapping from input sentence with words $S = w_0w_1\ldots w_n$ to its dependency tree graph $G$

Constraints (usually):
- Only one word dependent of root
- No cycles a ←b →a
- Can arrows be non-pojective or not?

**Projectivity**
- Projective parse: no crossing dependency arcs when words laid out linearly
- Dependencies for CFG tree definitionally are projective
- Usualy allow for non-projective structure

Typical methods of dependency parsing
1. Dynamic Programming- $O(n)^3$ 
2. Graph algorithms
3. Constraint satisfaction
4. Transition-based / determinitic dependency parsing
	1. Greedy choise of attachments guided by ml classifiers



**2 subproblems**
1. **Learning**: Given training set $D$ of sentences annotated with dependency graphs, induce a parsing model M that can be used to parse new sentences
2. **Parsing**: Given a parsing model M and sentence S, derive the optimal dependency graph D for S according to M.

**Transition based dependency parsing**

*Key idea*: Use state machine which defines possible transitions
- The larning problem: induce model 
- The parsing problem: construct optimal equnce of tranitions, given previously induced model

**Greedy deterministic tranition based parsing**

Transition system
- state machine (states + transitions)

**States**: for sentence $S = w_0w_1 \ldots w_n$, state described with $c = (\sigma, \beta, A)$
1. $\sigma$ stack of words $w_i$ from S
	1. LIFO
	2. Those that have already been read- being considered for forming relationships
2. $\beta$ buffer of words $w_i$ from S
	1. FIFO
	2. Remaining input word
3. Dependency arcs set $A$ of the form $(w_i, r, w_j)$ 
	1. w: word
	2. r: relation

For any sentenec
1. Initial state $c_0$ is of the form $([w_0]_\sigma, [w_1,\ldots , w_n]\beta, \emptyset)$
	1. Only the root is on the stack $\sigma$
2. Terminal state has form $(\sigma, []_\beta, A)$

Transition type between states
1. SHIFT: Move first word in the buffer to top of the stack
	1. Precondition: buffer nonempty
2. LEFT-ARC$_r$: Add dependency arc $(w_j, r, w_i)$ to the arc set A, where $w_i$ and $w_j$ are respectively second and first top of the stack. Remove $w_i$ from the stack
	1. Preconditions: >2 in stack and $w_i$ not the root
3. RIGHT-ARC$_r$: Add dependency arc $(w_i, r, w_j)$ to the arc set A, where $w_i$ and $w_j$ are respectively second and first top of the stack. Remove $w_i$ from the stack
	1. Precondition: >2 in stack

Think about example sequence


**Key question** how do you choose the next action?
- Not clear what dependency arc to assign, or when to shift instead

**Answer** use a dicriminative classifier (e.g. softmax)

In simplest form, no search
- VERY fast, linear time

**Greedy Transition-based Neural Dependency Parsing**

**Goal of model**: predict transition sequence from some intiial configuration to a terimnal configuration
Since greedy
- Tries to predict one transition at a time, based on features from current configuration

**Feature selection**

What should the input to neural network be?



Generally features for a given sentence S are some subset of
- $S_{\text{word}}$: vector representation of some of the words in S (and dependents)
	- e.g. top 3 word on stack and buffer, and first+second left/rightmost children of top 2 words
- $S_{\text{tag}}$: POS tag (from small discrete set)
- $S_\text{label}$: arc label for some of the words in S (from small discrete set)


*Conventional feature selection*
- Categorical indicator features: use a bunch of feature temapltes and fill out a binary, sparse vector
- Not efficient at all!

Instead, learn a dense + compact feature representation
- Use vectors to represent
	- Words
	- POS tags
	- Dependency labels

To get features
- Extract tokens by using the buffer and stack positions
- Contatenate vector representation of features- ie. words, POS, and dependencies

**Why are neural dependency parsers better?**
- Distributed representations
- Non-linear classifiers


General architectures
- Stack + buffer -> Input Layer X (lookup in matrix + contatenate) -> Hidden Lyaer -> Output Layer (softmax)

