---
title: Natural Language Generation
---
# NLG
NLP = NLU + NLG

NLG: producing language output for human consumption
Examples
- Translation systems
- Digitla assistant systems
- Summarization
- Creative stories
- Visual Description
- ChatGPT

Can classify by Open-endedness
- One end: machine translation
- Other: creative stories

Open-ended generation: output distribution has high freedom
- ie- higher entropy
# Review: Neural NLG + Training

The basis of NLG (review of lecture 5)
- Autoregressive text generation: each time step takes in sequence of tokens + outputs new one $\hat y_t$ 
	- Given model $f (\cdot)$ and vocab $V$, get scores $S = f(\{y_{<t}\},\theta)\in R^V$
- Non-open-ended tasks (e.g. MT): encoder-decoder system
	- Decoder: Autoregressive model
	- Encoder :Bidirectional encoder
- Open-ended tasks (e.g.story generation): typically just decdoer
- Training
	- Maximize probability of next token given preceding words
		- $L = - \sum_t\ \text{log}\ P(y^*_t|\{y^*\}_{<t})$
		- Remember “Teacher forcing”- reset each time step to ground truth
- Inference time
	- Decoder selects tkoen from the distribution (g = decoding algo)
	- $\hat y_t = g(P(y_t|\{y_{<t}\}))$ 

2 avenues to improve
1. Decoding
2. Training
# Decoding from NLG Models
What is decoding?
- Each $t$, model computes vectors of scores fore ach token in vocab
	- $S = f(\{y_{<t}\})$
- Then, compute Prob distribution $P$ over scores with softmax function
- Decoding algorihtm defines function to select token from distribution
	-  $\hat y_t = g(P(y_t|\{y_{<t}\}))$ 

How to find most likely string?
- Greedy Decoding: Select highest probability token in $P(y_t | y_{<t})$
- Beam Search: also maximizing log prob but wider exploration of candidates

**How to reduce repetition?**
- Simple: don’t repeat n-grams
- More complex:
	- Different **training** objective
		- Unlikelihood objective: penalize generation of already-seen tokens
		- Coverage loss: prevent attneiton mechanism from attneding to same words
	- Different **decoding** objetive
		- Contrastive decoding

For open ended generation, finding most likely string not super reasonable
- **Need to get random**!

**Sampling** to the rescue: sample from distribution of tokens rather than getting argmax
- Top-K sampling: (vanilla sampling has heavy tail of unreasonable answers)
	- Only sample from top k toens in prob distribution
	- Issue: can cut off too quickly! Or too slowly!
	- Solution: Top-p sampling
- Top-P (nucleus) sampling
	- Problem: prob distirbutions are dynamic (+ want differing ks)
	- Solution: sample from tokens in top $p$ cumulative probability mass (this varies the k)
- Typical Sampling
	- Rewight score based on entropy
- Epsilon Sampling
	- Set thrwshold for lower bounding valid probabilities

Still for randomness- **Temperature**
- Temperature hyperpamater $\tau$ to softmax to rebalance $P_t$
	- Within each exp, divide by $\tau$ 
- High temperature, more uniform $P_t$
- Opposite is true

**Re-ranking**
- Problem: what if decode a bad sequence from my model?
- Decoder number of candidate sequences
- Re-rank by a score that approximates qulity of sequences
	- E.g. Perplexity
	- Other: Style, discourse, entailment, logic consitency
	- Can compose multiple together

Takeaways
- Decoding still a challenge
- Decoding can inject bias
- Most impactful have been simple but effective modifcations to decoding algorihtms

# Training NLG Models

**Exposure Bias**
- Generation time: model’s inputs are previously-decoded tokens
Solutions
- Scheduling sampling: with probability $p$, decode token and feed as next input, instead of gold token
	- Increase $p$ over training
	- Can lead to strange training objectives
- Dataset Aggregation (DAgger):
	- At various intervals, generate sequences + add to training set as examples
- Retrieval Augmentation
	- Learn to retrieve seuqnece from corpus of human written prototypes- learn to edit sequence
- RL: cast text generatio as MDP
	- $s$ model’s representation of context
	- $a$ words which can be generated
	- $\pi$ decoder
	- $r$ provided externally

**Takeaways**
- Teacher Forcing is still main algorithm for training text generation models
- **Exposure bias** causes text gen models to lose coherence easily
- Rl can help 

# Evaluating NLG Systems

**Content Overlap Metrics**
Score of lexical similarity between generated and gold-standard
- Fast + widely used

N gram overlap metrics
- Not ideal for machine translation
- Progressively much worse for open ended tasks
	- Worse for summarization
	- Much worse for diaogue
	- Much much worse for story generation

**Model based metrics**
- Use **learned representations** of words/sentences
- Using **embeddings**: No more n-gram bottlenecks
- Word Distance functions
	- Vector Similarity: semantic distance
	- Word Mover’s Distance: word embedding simalirty matching
	- BERTSCORE: Pre-trained conextual embeddings from BET + matches words in sentences by cosing similarity
- Beyond Word Matching
	- Sentence Movers Similarity: Word movers Distance but text in continuous space using sentence embeddings
	- BLEURT: Regression model based on BERT

**MAUVE**
- For evaluating open-ended text-generation
- Computes information divergence in quantized embedding space

**Human evaluations**
- Gold standard in deeloping new automaatic metrics (new metrics must correlate with human eval)
- Basically- ask humans to evaluate quality of generated text on a dimension
- Issues:
	- Slow
	- Expensive
	- Can be inconsistent, irreproducible, illogical



