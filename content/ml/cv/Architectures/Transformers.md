---
title: Transformers
---



(Rest of Transfomer info in Fundamental ML Folder)


[Finish here](https://www.youtube.com/watch?v=QQY24LLww1A)


Basics: we know we can tokenize words, how to tokenize images?
- Image → list of patches
- Then, linear transformation (with convolution) → embedding
- Then, treat exactly the same as language

Vision size interpretation
- ViT-L/16 (large, with patch size of 16)
	- Lower patch size, many more tokens


Challenge: applying ViT to high-res images is hard

# Efficiency + acceleration techniques
Issue to resolve: compute grows quadratically with resolution
## Window Attention
Window attention- attention within a window (path)
Can graudally downsample the feature map size- so number of patches decreases
But
- Problem: Information only flows within map
- Solution (in Swin transformer): add *shift operation*- shift pixels so adjacent windows can let info propagate

Sparse window attention  (3..g for a 3d Point cloud which is quite sparse)
- Equal-window grouping: breaks the computaionl regularity (imablanaced + hard to parallelize)
- Instead: Equal sized grouping: ensures a balnced computation workload
## Linear Attention
Replace softamx attention with linear attention

## Sparse Attention

# Self-supervised learning for ViT

## Contrastive Learning
## Masked Image Modeling

# Multi-Modal LLMs

## Cross Attention (Flamingo)

## Visual Tokens (PaLM-E)





Generally
- ViT models-
	- When pretrained on large dataset, beats CNN based SOTA
	- ResNets better with smaller datasets
- Convolutional inductive bias is useful for small datasetes
- Learning directly from data is better for large ones

