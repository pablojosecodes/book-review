---
title: Word Vectors
---
**How do you represent the meaning of a word?** Think about it for a second. What actually captures what a **word** means.

Here are some approaches
- Denotational semantics
- Vectors from annotated discrete properties- use specific relationships like synonyms and hypernyms (“is a” relationships)
	- Issues: miss nuance, labor-heavy, no concept of word similarity
	- Example: WordNet
- Discrete Symbols- localist representation (eg. one-hot encoding)
	- Issue: no notion of similairty
- Distributional Semantics- meaning given by company it keeps
	- Example: Word2Vec

We tend to find distributional semantics most useful, especially in a deep learning paradigm.
## Word2Vec (Distributional Semantics)
Goal: optimize word vectors s.t. similarity represents probability of word given other word

2 models
- Skip gram (presented here)
- Continuous bag of words

Let’s formalize the **objective function**:
- Goal: For each position $t$, predict context words within a window f fixed sized m given center word $w_t$
- Definitions:
	- Likelihood $L(\theta) = \prod_t \prod_{j \text{ in context window}} P(w_{t+j}|w_t; \theta)$ 
	- $\theta$: variables to be optimized
- Objective function- average negative log likelihood
	- $J(\theta) = -\frac{1}{T}\text{log}L(\theta)$
- Thus, in order to minimize our object function, we must maximize our predictive accuracy (this is what we wanted!)

Question becomes- how do you calculate $P(w_{t+j}|w_t; \theta)$?
- use 2 vectors per word
	- $v_w$ when w is a center word
	- $u_w$ when w is a context word
- And make $P(o|c) = \frac{\text{exp}(u^T_ov_c)}{\sum_{w \in V} \text{exp}(u^T_wv_c)}$  (basic softmax)
	- Maps arbitrary values to probability distribution

Now we know enough to optimize, but how do you perform gradient descent?
-  $\theta \leftarrow \theta - \alpha \nabla_\theta J(\theta))$

**Issue with basic gradient descent**
- $J(\theta)$ defined as function of all windows in the corpus
- instead, we basically always use SGD

So we need to take derivative to work out the minimum

So, take derivative of the log likelihood
- $\frac{\delta}{\delta v_c}\text{log}\frac{\text{exp}(u_o^Tv_c)}{\sum \text{exp}(u_wv_c)}$ which becomes 2
	- $\frac{\delta}{\delta v_c} \text{log exp}(u_o^tv_c)$
	- - $\frac{\delta}{\delta v_c}\text{log}\sum\text{exp}(u_w^Tv_c)$  

Take $\frac{\delta}{\delta v_c} \text{log exp}(u_o^tv_c) = \frac{\delta}{\delta v_c} (u_o^tv_c) = u_o$

$\frac{\delta}{\delta v_c}\text{log}\sum\text{exp}(u_w^Tv_c)$ 
- Use chain rule
Etc.

### Skip-Gram Negative Sampling
Core idea: train binary logistic regression to differentiate true pair  (word + context) vs. noise pairs (word + random)

Why?
- Naive softmax is expensive (need to go over all of vocabulary)

Modify the loss function to take into account this new goal
- $J_{\text{neg}-\text{sample}}(u_o,v_c,U) = -\text{log}\sigma(u_o^Tv_c) - \sum_{\text{k sampled words} } \text{log}\sigma(-u_k^Tv_c)$ 
	- k negative samples
	- Maximize probability real outside word appears, minimize random word appearing

## GloVe
Key idea: connect count based linear algebra based modelsl ike (COALS) with direct prediction models like Skip-gram

Crucial insight: ratios of co-occurence probabilities can encode meaning components
- IE- $\frac{p(x|\text{ice})}{p(x|\text{steam})}$ means something  and we should encode it somehow
But how?
- Change the loss function
	- We want dot product to be similar to the log of the co-occurence
	- $J = \sum f(X_{ij})(w^T_i \tilde{w_j} + b_i + \tilde{b}_j - \text{log} X_{ij})^2$ 



## Evaluating word vectors
Generally, two methods
- Intrinsic
	- Evaluate on subtask
	- Fast to compute
- Extrinsic
	- Evaluate on real task

Intrinsic word vector evaluation
- Word vector analogies
	- Evalute by how well cosine distance after addition captures intuitive semantic/syntactic analogy questions
- Compare with human judgements of similarity
	- Tiger and cat how similar? Compare to human evaluations
**Extrinsic word evauation**
- Named entity recognition (look at if help)
	- identify references to a person organization or location
	- Retraining
		- We have trained word vector by optimizing over simpler intrinsic task, but can retrain on new task. 
		- Risky though!  Only do if training set is large

Softmax classification + regularization
- Remember the cross entropy loss of probability of word vector x being in class in j
	- $- \sum_i \text{log}\ \text{softmax}(p(y))$



## Co-Occurence Matrix
Key idea: build up a table of co-occurence once, rather than iterating through entire corpus, possibly multiple times

2 options
- Window based
- Full document
how doe it work?
Simple, just a symmetric table with time that word (one hot encoded kinda) has appeared in context window
Then, take the vectors (ie. columns, rows) that have been built up and use as word vectors
Issues?
- High dimensional, sparse, expensive
How to fix?
- Reduce dimenionality
**Dimensionality reduction techniques exit**

# Singular Value Decomposition

Hyperspace- space with >3 dimenions
Orthonormal- unit length vectors which are orthogonal
Gram-Schmidt Orthonormalization process
- Set of vectors → Set of orthonormal vectors
- Normalize firt vector + iteratively rewrite remaining vectors in terms of themselves minus multiplication of already normalized vectors
	- In essence
	- Normalize v1 (vector 1) to get n1
	- Assign w2 =  v2 - n1 $\cdot$ v2 * n1
	- Then, n2 = normalization of w2
	- Do the same for v3 using n1 and n2
		- To get $w_k$, $w_k = v_k - \sum_{k-1}u_t \cdot v_k * u_t$ 
		- Which we normalize to get $n_k$

Matrice
- Orthogonal- If $AA^T=A^TA=I$ 
- Diagonal- zero except for diagnoal

Eignevector
- Nonzero vector which statisfies $A \overrightarrow{v} = \lambda \overrightarrow{v}$
	- $A$ square matrix
	- $\lambda$ scalar (eigenvalue)
	- $\overrightarrow{v}$ eigenvector
- Can solve for them with a system of linear equations

**Singular value decomposition**
**Core idea**: take high-d highly variable set of data points, reduce to a lower dimenional space which exposes substructure of original data more clearly
3 ways to view SVD
1. Correlated variable → uncorrelated variable
2. Identifying + ordering dimensions along which data points exchibit the most variation
3. Find best approximation of the original data points with fewer dimensions (data reduction)

**Based on Theorem from Linear algebra: $A_{mn}=U_{mm}S_{mn}V_{nn}^T$**
- $U^TU = I$
- $V^TV = I$

**Core idea**: you can represent a rectangular matrix A into the product of three matrices
- Orthogonal matrix U
- Diagonal matrix S
- Tarnspose of orthogonal matrix V
