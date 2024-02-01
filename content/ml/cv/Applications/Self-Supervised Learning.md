---
title: Self-Supervised Learning
---
Goa: learn from data withouot manual label annotation
- Solve “pretext” tasks which produce good features for downstream tasks

Some pretext tasks
1. Predict image transformations
	1. Rotation prediction
	2. Jigsaw puzzle
2. Complete corrupted images
	1. Image complextion
	2. Colorization

Evaluation
- Don’t actually care about performance of the self-supervised tasks
- Evaluate on downstream target tasks instead

Thus, the steps are
1. Learn good feature extractors from self-supervised pretext tasks
2. Attach shallow netowrk to feature extractor
	1. Train this shallow network on target task with small amount of data

# Pretext Tasks from Image Transformations

**Pretext task: Predict Rotations**

Idea: model can recognize correct rotation of an object if it has “visual commensense”

Learns classification
- 90 / 180 / 270 / 0

**Pretext task: predict relative patch locations**
Given center patch and another patch from its context, predict where in grid the second patch comes from

**Pretext task: solve jigsaw puzzles**
Given grid of 9 patches, order

**Pretext task: image coloring**

**Pretext task: video coloring**

Idea: model the temporal cohrrence of colors in videos (eg car continues to be red)

Hypothesis: learning to color video frmaes should allo wmodel to track regions or objects without labels

Learning objective: establish mappings between reference and target frames in a learned feature space

Learning
- Attention map on the reference frame $A_{ij} = \text{softmax}(\text{feature})$ 
- Predicted color: $y_j = \sum_i A_{ij}c_i$ 
	- Weighted sum of the reference color

Colorization → tracking
- Use the leraned attentino to
	- Propagate segmenttion masks
	- Propagate pose keypoints

Key points
- Pretext tasks focus on “visual common sense”
- Models forced to learn good features
- Don’t care about performance on pretext tasks
- Problems
	- Coming up with individual pretext tasks is tedious
	- Learnings may not be general
- More general pretext task?
	- Contrastive representation learning

# Contrastive Representation Learning
Key idea: score$(f(x),f(x)^+)$ >> score$(f(x),f(x^-))$
- x: reference sample
- $x^+$: positive sample
	- Can be transformed with pretext tasks?
- $x^-$: negative samle

Given score function
- Learn **encoder function** which yeields high score for $(x,x^+)$ and low scores for negative pairs $(x,x^-)$

Loss function (given 1 positive sample and N-1 negative samples)
- $L = -E_x[\text{log}\frac{\text{exp}(s(f(x)),f(x^+))}{\text{exp}(s(f(x),f(x^+))\ +\ \sum_{j=1}\ \text{exp}(s(f(x),f(x^-+j))}]$  
- Known as the InfoNCE Loss
- Lower bound on the mutual info between $f(x)$ and $f(x^+)$ 
	- $MI[f(x),f(x^+)]-\text{log}(N)\geq -L$ 

**SimCLR**: Basic framework for Contrastive Learning
- Use cosine similarity as the score function
- Use projection network $\textbf{g}(\cdot)$
	- Project features to pspace where contrastive learning is applied
- Positive samples from data aug
	- Cropping, color distortion, blur
- Mini-batch training
- To use for downstream applicatoins
	- Train feature encoder on large dataset (eg imagenet) using SimCLR
	- Then- freeze feature encoder + train linear classificaiton (or other purpose) on top with labeled data
- Design choices for SimCLR
	- Projection head: linear / non-linear projection heads improve representation learning
		- Why?
		- Maybe-
			- The contrastive learning objective discards useful information, and representation space $\textbf{z}$ is trained to be invariant to data transformation
			- Projection head may let more info be preserved in the $\textbf{h}$ representation space
	- Large batch size: Crucial!

Pseudocode

for given minibatch
- for all instances
	- Generative positive pairs by sampling data augmentation functions
	- Iterate through and use each of the 2N samples as reference, compute average loss
		- InfoNCE loss


**Momentum Contrastive Learning (MoCo)**

Differences to SimCLR:
- Running queu of negative samples (keys)
- Computes gradients + updates encoder only through the queries
- Decouples mini-batch size from number of keys
	- Can have large number of negative samples
- Key encoder slowly progresses using momentum update rules
	- $\theta_k \leftarrow m\theta_k + (1 - m)\theta_q$

**Moco V2**

Hybrid of SimCLR + MoCo
- **SimCLR**: non-linear projection head + strong data aug
- **MoC**: momentum-updated queus which alow training on large number of negative samples

When comparing the 3 some takeaways stand out
- nonlinear projection head and strong dat aug are **crucial** for constrastiv elearning
- Decoupling mini-batch size w/ negative sampling size lets Moco-V2 outperform SimCLR with smaller batches
	- with smaller memory footprint


## Instance vs Sequence Contrastive Learning

Instance-level
- Positve / negative instances
- e.g. SimCLR, MoCo

Sequence level
- Sequential / temporal orders
- e.g. Contrastive Predictive Coding


**Contrastive Predictive Coding**
term by term
- Contrastive: right vs wrong sequneces
- Predictive: predicts future patterns given current context
- Coding: learns feature vecotrs (ie code) for downstream tasks

Steps
1. **Encode** all samples in sequence to vectors $z_t$
	1. $z_t = g_{\text{enc}}(x_t)$
2. **Summarize** the context into context code $c_t$
	1. Use an auto-regressive model $(g_{\text{ar}})$ 
	2. Original paper uses GRU-RNN
3. **Get loss** between context $c_t$ and future code $z_{t+k}$ using time-dependent score function
	1. $W_k$ trainable matrix
	2. Loss: $s_k(z_{t+k},c_t) = z^T_{t+k}W_kc_t$

Example use case
- Speaker classification


# Takeaways-  contrastive representation learning

General formulation: score$(f(x),f(x)^+)$ >> score$(f(x),f(x^-))$

InfoNCE loss: n-way classification between positive / negative samples
- $L = -E_x[\text{log}\frac{\text{exp}(s(f(x)),f(x^+))}{\text{exp}(s(f(x),f(x^+))\ +\ \sum_{j=1}\ \text{exp}(s(f(x),f(x^-_j))}]$  

**SimCLR**: framework for crl
- Non-linear projection head: flexible learning
- Simple + effective
- Large memory footprint

**MoCo**: contrastive learning w/ momentum sample encoder
- Decouple negative sample size from minibatch size with queue

**MoCo v2**
- Combines nonlinear projection head, strong data aug, w/ momentum contrastive learning

**CPC**: sequence level contrastive learning
- Right vs wrong squence
- InfoNCE loss w/ time dependent score function

Some other examples to think about
- CLIP: contastive learning between image + natural language
- Dense Object Net: contastive learning on pixel-wise feature descriptors
