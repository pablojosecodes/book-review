---
title: Code Generation
---


**Program Synthesis**
Programs that write programs

Program synthesizer: high level specification → program
What specification?
- Logical formula
	- Non-trivial- might be harder than writing program itself
- Equivalent program (e.g. slower)
- Input/output examples
	- E.. flashfill
	- Ambiguous
- Natural language description
Why specification?
- When easier to specifiy what than how program should do 

Main challenges
- infinite space of programs
- specifications are ambiguous- how to capture human preference?


**As Pragmatic Communication**

Ambiguity
- In language, not a bug- feature of efficiecny
- Human communication depends on cooperativity
	- Assume partner being as informative as possible
- Use this in context to perform *pragmatic reasoning*

RSA (Rational Speech Act): Bayesian model of how we choose or interpret uttace given context by recursively reasoning about other party
- Given utterance u, 
	- Literal listener (L0) would assign $P_{L0}(o|u) \propto [[u]](o) P(o)$
	- Pragmatic Speaker which wants to refer to o, chooses u reasoning about Lo, balancing
		- How likely L0 is to infer o given u
		- How costly u is 
		- Thus, $P_{S1}(u | s) \propto \text{exp}(\text{alpha}((\text{log}P_{L0}(o | u) - \text{Cost}(u)))$ 
	- Pragmatic listener interprets u asuming S1 is speaking
		- $P_{L1}(o | u) = P_{S1}(u | o) P(o)$
- Could keep recursing, but usually S1-L1 pair is enough


In program synthesis
- Utterance = specification
- Synthesizer = listener, trying to infer what program we’re referring to 

Assuming finite set of specifications and programs, could build. meaning matrix
- $M[s][p]$ = 1 if program $p$ satisfies $s$

Obvious issue: only works for finite pgorams / specification spaces


**with language models**

LMs can generate code

CodeX core idea: train llm on majority code data

Codegen eval for LMs
- Synthesis challenge: given docstring, implement (new dataset)

Sampling vs temperature
- Sampling more increases chance of getting one right
- Low temperature: higher likelihood, less diversity
- Higher temperature: lower likelihoods, more diversity

Ranking (don’t want to present 100 samples)
- #1: sample small number
- #2: rerank large number, show subset
- ORACLE: run on hidden tests

AlphaCode
- Encoder-decoder transformer
- Multi-query attnetion, not full multi-head attnetion blocks
- Pipeline
	- Pretraining: Standard cross entropy loss on lots of github code
	- Finetuning: 
		- RL fine-tuning on GOLD
		- Value conditioning: incorporate incorrect submissions- prepend comment on whether accepted 
			- In theory, makes good use of the limited data- still incorporates something about how to write code
	- Sampling: up to 100k
		- Filtering: discord those which fail public tests
		- Clustering
			- Separate model to generate sample inputs
			- Cluster solutions with similar outputs, return from the 10 most common program clusters

**Programs as tools**

Toolformer, etc. to get language models to use tools
Toolformer
- First approach: use claculator by having annotations in training dataset
- Second approach: few shot prompting
- Toolformer: self-supervised approach to teach models new tools
	- Few examples of each tool + larger dataset for task without tool use
	- Use in context leraning to insert candidate api calls in training examples
	- Call APIs- evaluate whether resilt decreases perplexity of the rest of the solution
	- Fine tune model on cases where it does
	- Result: can now often use APIs