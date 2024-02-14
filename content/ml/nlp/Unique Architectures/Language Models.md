---
title: RNNs
---

Language models
- Goal: compute probability of occurence of a number of words in a particular sequence
Why do we care about language modeling?
- Benchmark task that helps measure progress on understanding language (useful for much more in recent times, presumably)
Denotation:
- Probability of sequence of m words $\{w_1,\ldots, w_m\}$ denoted as $P(w_,1\ldots, w_m)$
- Usually, condition on a window of $n$ previous words, not all previous words
	- $\prod_i P(w_i|w_{i-n},\ldots, w_{i-1}])$ 

# n-gram language models
**Core idea**: sample from distribution based on conditional probabilities, holding the Markov assumption
- Count of each n-gram compared against frequency of each word
Example: if model take bigrams
- Get the frequeency of each bi-gram by combining word with its previous word
- Divide by frequency of corresponding unigram.

Bigram
- $p(w_2|w_1) = \frac{\textit{count}(w_1,w_2)}{\textit{count}(w_1)}$
- $p(w_3|w_1, w_2) = \frac{\textit{count}(w_1,w_2, w_3)}{\textit{count}(w_1, w_2)}$

Get probabilitiy distribution, and sample from them

**Makov assumption**: assume that word only depends on preceding $n-1$ words
- Not actually true in practice
- Lets us use the defintion of conditional probability

### Main issues with n-gram language models
**Sparsity**: 
1. If the trigram never appears together, probability of third term is 0.
	1. Fix: *smoothing*, add a small $\delta$ to count for each word
2. If denominator never occurred together in the corpus, no probability can be calcualtd
	1. Fix *backoff*, condition on smaller $n$
**Storage**: 
1. As $n$ or corpus size increase, model size increase as well


# Neural language model? 
How to build?
- Naive: fixed window-based neural model, softmax over vocab
Why is this better than n-gram? 
- Distributed represntations instead of the very sparse representations of word sequnces in n-gram
	- In theory, semantically similar words should have similar probabilities
- No need to store all observed n-grams
Remaining issues
- Small fixed window
- Each word vector multiplied by completely different weights- no symmetry
Use RNNs!
‘


# RNNs
[Basics covered here](obsidian://open?vault=Obsidian%20Vault&file=Learning%2FFundamental%20ML%2F(A)%20Supervised%2FRNNs)

## RNN text generation
Just do repeated sampling
- Feed in start token
- Take the sampled word from given state, embed, feed into next timestep as $x_t$ 
- Until EOS token
Note: can do much more than just language modeling

**Pros**
- Variable input sequnce
- Model size doen’t increase for longer input sequences
- Computation for step t incorporates all prior knowledge (theoretically)
- Same weights at each timestep (symmetry)
**Cons**
- Slow- sequential computation

## RNN Translation Model

**Traditional translation model**: many ML pipelines

RNNs- much simpler
Basic functionality
- Hidden layer time-steps encdode foreign langauge words into word features
- Last time steps decode into new language word outputs

**Necessary extensions** to achieve high accuracy translation
1. Different RNN weights for encoding and decoding 
	1. Decouple the w- more accuracy prediction of each of the two RNN module
2. Compute hidden state using 3 inputs:
	1. Previous hidden state
	2. Last hidden layer of the encoder
	3. Previous predicted output word
3. Train RNN with multiple RNN layers
4. Train bi-directional encoders to improve accuracy
5. Train RNN with input tokens reverse



# Evaluating language models
Standard evaluation metric: perplexity
Perplexity: Geometric mean of inverse probability of corpus according to language model
- perplexity = $\prod_t(\frac{1}{P_\text{LM}(x^{(t+1)}|x^{(t)}, \ldots, x^{(1)})})^{(1/T)}$ 
Also equivalent to the exponential of the cross entropy loss