---
title: Coreference Resolution
---
# Coreference Resolution
Coreference resolution
- Identifying all mentions that refer to the same entity in the world

Applications
- Full text understanding
- Machine translation
- Dialogue systems

In 2 steps
- Detect mentions (easy)
- Cluster mentions (hard)

Mention detection
- Mention: text referring to an entity
- Either
	- Pronoun
	- Named Entity (people, place)
	- Noun phrases 
- To detect: usually pipeline o other NLP systems
	- Pronounce: POS tagger
	- Named Entities: NER System
	- Noun phrases: Use a (consistuency) parser

Marking all of the 3 types over-generates mentions
- Example: “It is sunny”- it isn’t a mention
How to deal with these bad mentions
- Naive: classifier filter out 
- More common: keep mentions as candidate mentions


Some linguisitics
- Coreference: 2 mentions refer to same entity in the world
- Anaphora: When term (anaphora) refers to other term (antecedent)
- Pronomial Anaphora: when anaphora is coreferential
- Bridging Anaphora: Not all anaphoric relations are coreferential! 
	- Example: We saw a concert last night; the tickets were reall expensie
- Cataphora: Reverse order of anaphora

# Coreference Resolution Models


**Rule-based (Hobbs)- pronomial anaphora resolution**
Messy naive algorithm that worked very long
1. Begin at NP immediately dominating pronoun
2. Go up tree to first NP or S. Call this X, and the path p.
3. Traverse all branches below X to the left of p, left-to-right, breadth-first. Propose as antecedent any NP that has a NP or S between it and X
4. If X is the highest S in the sentence, traverse the parse trees of the previous sentences in the order of recency. Traverse each tree left-to-right, breadth first. When an NP is encountered, propose as antecedent. If X not the highest node, go to step 5. 
5. From node X, go up the tree to the first NP or S. Call it X, and the path p.
6. If X is an NP and the path p to X came from a non-head phrase of X (a specifier or adjunct, such as a possessive, PP, apposition, or relative clause), propose X as antecedent (The original said “did not pass through the N’ that X immediately dominates”, but the Penn Treebank grammar lacks N’ nodes….)
7. Traverse all branches below X to the left of the path, in a left-to-right, breadth firstmanner. Propose any NP encountered as the antecedent
8. If X is an S node, traverse all branches of X to the right of the path but do not go below any NP or S encountered. Propose any NP as the antecedent.
9. Go to step 4 

Wino-grad schema: knowledge-based proomial coreference
- Example: The city council refused the women a permit because they advocated violence. (what is “they” coreferential to?)
- Alternative to Turing test 


**Mention pair coreference models**
Train binary classifier which assigned each pair- probability of being coreferent $p(m_i, m_j)$
- e.g. given “she” look at all candidate antecedents

Training: 
- N mentions in document
- $y_{ij} = 1$ if mentions $m_i$ and $m_j$ are coreference- else -1
- Loss: cross-entropy
	- $J = \sum_{i=2}^N \sum_{j=1}^i y_{ij}\ \text{log}\ p(m_j, m_i)$

Test time 
- Pick threshold and add coference links when $p(m_i,m_j)$
- Can fill in gaps by using things like transitivity

Cons
- Given long document, many mentions only have one clear antecedent, but asking moel to predict all of them
- Solution: mention ranking- predict only one antecedent / mention

**Mention-pair and mention-ranking models**
Idea: assign eaeh mention highest scoring candidate antecedent according to model
- Use dummy “NA” mention to decline linking to anything

Apply softmax over scores for candidate antecedent 

Training
- Maximize  the following
- $\sum_j^{i-1}\ \mathbf{1}(y_{ij}=1)p(m_j, m_i)$
	- Iteration through candidate antedents (previously ocrring mentions)
	- For ones that are coreferent to $m_j$ we want a high probability

But, how to compute probabilities?
1. Non-neural statistical classifier
2. simple neural network
3. More advanced model using LSTMs, attention, transformers

**(1) Non-neural coref model**
Features
- Person/number//gender agreement
- Semantic Compatibility
- Some syntanctic constraints
- More recently mentioned entities preferred
- Prefers entities in subject position
- Paralelism
- $\ldots$

**(2) Neural coref model**
Standard FFNN
- Input: word embeddings + categorical features
	- Embeddings: prev two words, first word,last word, head word, of each mention
	- Other feautres: ditance, document genre, speaker info

(3) End-to-end model
Improvmenets
- LSTM
- Attention
- Mention detection + coreference end-to-end
	- No mention detection step!-
	- Instead, consider ever span of text (with length limit) a candidate mention
Steps
1. Embd words
2. Run bidirectional LSTM over doc
3. Represesnt each span of text as vector
	1. Span representation: $g_i = [x^*_{\text{START}(i)},x^*_{\text{END}(i)},\hat x_i, \phi(i)]$ 
	2. Explanation of terms: 
		1. Hidden states for spans’ start and end
			1. Represents context to left anad right of span
		2. Attention based representation of words in span
			1. Span itself
		3. Additional feautres
			1. Info not in text
4. $\hat x_i$ attention-weighted acerage of word embedings in the span
	1. $\alpha_t = w_\alpha \cdot \text{FFNN}_\alpha (x^*_t)$ 
	2. Attention distribution $a_{i,t}$ is osoftmax over attention scores for the span
	3. Then weight $\hat x$ using $x$ and $a$ 
5. Finally- score each pair of spant o decide if coreferent mentions
	1. $s(i,j) = s_m(i) + s_m(j) + s_a(i,j)$
	2. Explanation of terms: 
		1. Are they coreferent?
		2. Is i a mention
		3. Is j a mention
		4. Do they look coreferent?

**Transformer-based coref (now SOTA)**
 Can learn long-distance dependencies
- (Idea 1) SpanBERT: pretrain BERT to be better at spna-based prediction task
 - (Idea 2) BERT-QA: treat Coreference like QA task
 - (Idea 3) maybe no need for spans and can represent mention with a word and make things $O(T^2)$



 # Evaluation and current results
 Coreference evaluation: many different metrics
- Usually report average of a few

Intuition: metrics think of corefernce as clustering + evaluate quality of the clusters


