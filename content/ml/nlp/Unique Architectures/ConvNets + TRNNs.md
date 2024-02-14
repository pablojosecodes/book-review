---
title: ConvNets + TRNNs
---
# CNN review

RNNs can’t capture phrases without prefix context
- Often capture too much of last word in final vector

Core CNN Idea: what if we compute vectors for every possible word subsequence of a certain length and hten group them afterwards?

**Convolution**: 
- 1d: $(f \times g)[n] = \sum_{m = -M}^M f[n - m ]g[m]$ 
- Classically used to extract image features
- Basically same as image convolutoin but for 1d and text- same ideas of stride, pool, and padding

# CNN for Sentence Classification

Goal: sentence classification

SIngle layer CNN for sentence classification
1. Sentence = concatenation of word vectors
2. Apply filters to concatenation of words in range $i:j$
3. Apply convolution filter to all possible windows
4. Compute feature for each CNN layer (length h)
	1. $c_i = f(w^Tx_{i:i+h-1} + b)$ 
5. Result: feature map
6. Can use pooling 
7. Final softmax layer


# CNN Potpourri
Growing Toolkit
1. **Bag of vectors**: suprisingly good baseline for classification
2. **Window Models**: Good for single word classification without wide context
3. **CNNs**: Good for classification- easy to parallelize
4. **RNNs**: Cognitively plausible, but not best for classification
5. **Transformers**: Basically best thing since slided bread

# Deep CNN for sentence classification
What happens if we build deep vision-like system for NLP?
- Working up from character level

**Example**- VD-CNN: kind of like VGG or ResNet
Each convolutional block
- 2 Conv Layers
- Btch norm
- ReLU



# Tree Recursive Neural Nets
Human language kind of recursive
- Think of recursion structure- term refers to term which refers to \_, etc. in a tree like structure

How to map phrases into vector space?
- Use principle of compositionality
- Meaning (vector) of phrase / sentence is determined by
	- Meanings of its words
	- Rules that combine them

**Recursive models**: jointly learn sparse trees and compositional vector representations
- Require tree structures

For structure prediction
- Input: 2 candidate childrens’s representations
- Output:
	- Semantic representation if two nerds are merged
	- Score of how plausible new node would be 
- 


# Recursive Neural Tensor Networks
**Idea**: allow both additive and mediated multiplicative interactions of vectors




