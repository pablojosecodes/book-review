---
title: Video Understanding
---

# Videos

Lots of video models

## Single Frame CNN
**Good baseline**

Core idea: just train a normal 2D CN to classify video frames independently!

## Late Fusion
**Intuition**: high level appearance of each frame + combine them

Run 2D CNN on each frame → concatenate features → feed to MLP

With pooling:  Run 2D CNN on each frame → pool features → feed to linear

Issue: hard to compare low level motion between frames
## Early Fusion

Compare frames (temporal dimension) in first conv layer, then standard 2D CNN
- Collapse all temporal info in first conv

No temporal shift invariance! Separate filters for same motion at different times

Isue: only one layer of emporal processing- may not be enough

## 3D CNN 
**Intuition**: 3d version of convolution + pooling to slowly fuse temporal info over the course of the network

Temporal shift invariant! Each filter slides over time

Ecah layer: 4D tensor: D x T x H x W 
- 3D Conv
- 3D Pooling

**C3D**: the VGG of 3D CNNs
3D CNN which uses all 3x3x3 conv and 2x2x2 pooling (also a Pool1 which is 1x2x2)

Still has issue- 3x3x3 Conv is very expensive

## Idea: optical flow
Measure motion 

Optical flow highlights lcoal motion
- Where each pixel will move in the next frame

## Two stream networks
Separating motion and appearance

2 inputs
- Stack of optical flow
- Single image

Modeling long-term temporal structure
- So far, temporal CNNs only model local motion- what about long-term structure?

Process local features using recurrent network

Can use multi layer RNN type structure to process videos

Recurrent convolutional network
- In normal 2D CNN: Input → (2d Conv) → Output features
- Recurrent CNN Features from Layer L, timestep L and features from Layer L-1, timestep T → (RNN-like recurrence) → feautres for layer L, timestep t

Issue: RNNs are slow for long seuqnces (not parallelizable)

Recall: different ways of processing sequences 
 - RNN: for ordered sequences (in video, CNN+RNN)
	 - Pros: Good for long sequences
	 - Cons: Not parallelizable
 - 1D Convolution: for multidimensional grids (in video: 3d Convolution)
	 - Pros: highly parllel
	 - Cons: Bad at long sequences
 - Self-Attention: for sets of vectors
	 - Pros: Good for local sequences, highly parallel
	 - Cons: Memory intensive

Spatio-Temporal Self-Attention TODO


Inflating 2D Networks to 3D (I3D)
- Already lots of work done on images, can we extend to video?
- Idea: take 2D CNN architecture + replace each 2D conv/pool with 3D version
- Can use weights to initialize 3D conv, copy in space and divide

Vision Transformers for Video
- Factorized Attention
- Pooling module


Visualizing Video Models
- 

There are more tasks than just classifying short clips

Temporal Action Localization
- Given long untrimmed video, identify frames corresponding to actions

Spatio-Temporal Detection
- Give long untrimmed video, dertect peoplel in space and time + classify activities