---
title: CNN History
---

**CNN Architectures- in search of depth**

# LeNet

# AlexNet: 
8 layers
First deep learning architecture to win at ImageNet
Key ideas
- First use of ReLU
- Heavy data augmentation
- Dropout regularization
- Ensembling of models
- GPUs- spread neItwork across 2

# VGG
Key ideas: smaller filters + deeper netwroks
- **Only** 3x3 filters stride 1, 2x2 maxpool stride 2
- As input space decreases, depth volume incerases
	- Double number of feature maps when spatial extend halves.
- Why 3x3?
	- Stacked conv. layers have a large receptive field
		- 2 3x3 = 5x5
		- 3 3x3 =  7x7
	- And 
		- More nonlinearity
		- Fewer paramters
- Data augmentation at training + testing time
	- Multi scale training
		- Randomly cropped ConvNet input
		- Different training image size
	- Standard jittering

# GoogLeNet
Goal: deeper networks with computational efficiency

**Key Ideas**
- 22 layers
- Replicated modular structure (Inception)
- Careful design to achieve computational savings (12x fewer parameters than AlexNet!)
- More depth, less parameters
- No FC layer

**Architecture**
- First, typical convs and pools
- Then, stacks of inception modules
	- Combine depthwise
		- 1x1 →
		- 1x1 → 3x3 →
		- 1x1 → 5x5 →
		- 3x3 pooling → 1x1 →
- Then, typical classifier- without FC
- Also, some auxiliar classification outputs

## Inception module: 
Key idea: exploit local correlations in parallel
- Parallel filter operations on input from previous layer
- Concatenate filter outputs depthwise

Naive implementation
- Issue: computationally expensive
	- Lots of operations if combining all these depthwise
- Solution: 1x1 convolutional layer “bottleneck” layers

**Intuition**: 
- Cover very local clusters with 1x1 convolutions
- + Cover more sprea out clusters with 3x3 convolutions
- + Cover more spread out clusters with 5x5 convolutions

**1x1 convolutional layer**
- Introduces bottlenckes
- Reduces dimensionality of feature maps

Inception V2, V3 Modules
- Key ideas
	- Batch normalizatoin
	- Improve convolution performance: (1x)-factorization, convolutoin stacking
	- Efficient grid size reduction
- Fundamental ideas common to inception nets
	- Net composed of replciated instances of a single module
	- Module composed of multiple branches
	- Module has “bottlenecks”

# **ResNet**
Fundamentals
- 152 layers
Background
- What happens with deeper and deeper layers on plain cnn?
	- Worse training and test
- Hypothesis: problem with deper layers is an optimization problem, deeper networks are harder to optimize

**Conceptual understanding**:
- Deep networks must be able to learn at least as well as shallow inputs 
	- They could, for example, learn up to same depth, and keep everything else identity function
- Key insight- don’t learn transformation itself, learn the residual

**Residual network fundamentals**
- As more layers, harder to train
- Learn the residual mapping instead of the underlying mapping directly.
- Residual connect optimizes to find $H(x) = F(x) + x$
	- $H(x)$: desired underlying mapping
	- $F(x)$: residual that the network layers are learning

**Architecture**
- Stack residual blocks 
- Residual block
	- 2 3x3 conv layers learning F(X)
	- x passes to beyond 2nd layer
- Periodically double # of filters and downsample using stride 2
- Deeper networks use bottleneck layers
	- Residual block for these
		- 1x1 conv, 64
		- 3x3 conv, 64
		- 1x1 conv, 256
**Training**
- Xavier intialization
- Batchnorm after every conv layer

# **Inception-ResNet**
Same philosophy of ResNet / GoogleNet 
- ResNet shortcuts 
- Inception multiple branches.

# **ResNeXt**
Same philosophy of ResNet / GoogleNet 
- ResNet shortcuts
- Inceptional multiple branches
- Grouped Convolutions

# **DenseNet**
Uses Dense Blocks TODO

DenseBlock
-  Each layer  is connected to every other layer in feedforward fashion

Properties
- Feature reuse
- Improved graident flow (direct access to gradients)
- Implicit regularization
Experimental results
- Increased performance with higher depth + Growth rate
- More paramter efficient

# **SE-Net**
- Improve representation
	- By explicitly modelling hannel relationships
Squeeze + Excitation Block
- Model channel interdependencies by
	- Squeeze global info in each channel
		- $z_c = F_{sq}(u_c) = \frac{1}{H \times W} \sum_{i =1}^H \sum_{j=1}^W u_c(i,j)$ 
	- Exitation by adaptive recalibration of channel importance
		- $s = F_{ex}(z,W) = \sigma(g(z,W)) =  \sigma(W_2\delta(W_1z))$ 
		- where
	- Re-scaling channels
		- $\textbf{x}_c = F_{scale}(u_c,s_c) = s_c \cdot u_c$

Depthwise separable Convolution (DwSC)
- Standard Convolutional layer
- Idea: reduce paramters and operatoins by
	- Decouple. number of filters from number of output feature maps
	- Compute convolution in 2 steps
		- Depthwise
			- Introduce one $D_K \times D_K$ filter per input feature map, $M$ features.
			- $\#$ of multiplications $D^2_K \times D^2_G \times M$ 
			- $\#$ of parameters $D^2_K \times M$
		- Pointwise
			- Introduce one $1 \times 1$ filter per output feature map, $N$ filters
			- $\#$ of multiplications $ D^2_G \times M \times N$ 
			- $\#$ of parameters $M \times N$	
	- Total number of multiplications/parameters: add them
	- Compared to standard convolution: 1/10 the number of multiplications / parameters


# **Mobile Net**
- Simple computatonal complexity control with the parameters:
	- $\alpha$: base network
	- width (# of channels)
	- $\rho$: base network resolutoin (input w x h)
	- Computatonal cost is  $D_K^2 \times \alpha M \times \rho^2 D_G^2 + \rho^2 D_G^2 \times \alpha M \times \alpha N \sim o(\alpha^2 \times \rho^2)$
- **Inverted residual with linear bottleck layer**

# **Xception**
Extreme inception
- Inception module with a large number of identical towers

# **Network Architecture Search (NAS)**
From feature engineering to architecture engineering
NAS automates architecture design
NASNET
- Discrete architecture search

# Efficient Net 
