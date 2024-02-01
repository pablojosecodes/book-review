---
title: Question Answering
---

# What is QA?

What info does system build on?
- Text, web documents, knowlege bases, …
Question type
- Factoid vs non-factoid, open-domain vs closed-domain, simple vs compositional, …
Answer type
- Short text segment, paragraph, list, yes/no, …

In deep learning era- almost all are built on pre-trained language models

Focus here is unstructured text anwers, but also exists
- Visual QA, ...

# Reading Comprehension
Core task: comprehend text + answer questions about its content

Why care?
- Practically useful
- Testbed for understanding language
- Many NLP can be reduced to reading comprehension
	- Information extraction
	- Semantic role labeling
	- ...

Dataset: SQuAD
- 100k (passage, question, answer) triplets
- Almost solved dataset
- **Evaluation**: exact match (0 or 1) and F1 (partial credit)
- Compare predicted answer to each of 3 gold answers, then take max scores
	- Take average all examples for both exact match and F1
	- Example: 
		- Gold answers: {left Graz, left Graz, left Graz and severed all relations with family}
		- Prediction: {left Graz and served}
		- Exact match = max{0,0,0} = 0
		- F1: max{0.67, 0.67, 0.61} = 0.67

Other QA datasets
- TriviaQA: trivia enthusiasts
- Natural Questions: from Google search
- HotpotQA: wikipedia questions- involve getting info from two pages


**Neural models for reading Comprehension-** 
How to solve SQuAD?

Problem 
- Input: $C = (c_1, \dots)$, $Q=(q_1, \ldots)$, $c_i, q_i \in V$ 
- Output: 1 ≤ start ≤ end ≤ N
	- Answer: span in passage

Models- 

LSTM based with attention
- Stanford Attentive reader
- (BiDAF) Bidirectional Attention Flow Model

BERT for reading comprehesion
- BERT: has 2 training objectives
	- MLM
	- NSP
- Incorporate loss
	- $L = -\text{log}\ p_{\text{start}}(s^*) - \text{log}\ p_\text{end}(e^*)$
	- Where $p$ is softmax of (weights $\cdot$ hidden vector of $c_i$)
- Works very well

Better pre-training objectives?
2 ideas (SpanBERT)
1. Masked contiguous spans of words
2. Use two end points of span to predict all masked words in between



# Open domain QA
Core difference: not given passage, instead given a large collection of docuemnts

More challenge + more practical

**Retriever-reader framework**
- Input: collection of documents and query
- Output: answer string A
- Retriever: $f(D, Q) \rightarrow P_1, \ldots, P_K$ (pre-defined k)
- Reader: $g(Q, \{P_1, \ldots, P_K\}) \rightarrow A$ (reading comprehension problem)

In DrQA
- Retriever = standard TF-IDF info-retrieval sparse model
- Reader- same neurla reading comprehension model we just learned

**Can also train the retriever**
- Join training
	- Each text passage encoded as vector using BERT + retrieve score measures as dot product between question representatinos and passage representations
- Dense passage retrieval: just rain retriever using question-answer pairs

LLM can also do open-domain QA without explicit retriever stage