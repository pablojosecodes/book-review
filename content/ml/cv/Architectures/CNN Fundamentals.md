---
title: CNN Fundamentals
---

**Fundamental ideas**
- 2D/3D Spatial Distribution
- Local/Sparse interactions
- Receptive field
- Parameter Sharing
- Spatial Pooling


## Fun Techniques

#### Batchnorm
**Core idea**: force inputs to be “nicely scaled” at each layer
- Usually inserted after FC or CONV layers, before noninearity

Consider a batch of activations at some layer
- Get mean and deviation
	- $\mu_j = 1/N \sum_i x_{i,j}$
	- $\sigma^2_j = 1/N \sum_i (x_{i,j}-\mu_j)^2$
- Learn $\gamma$ and $\beta$

$\hat{x}^{(k)} = \frac{x^{(k)} - E[x^{(k)}]}{\sqrt{\text{Var}[x{(k)}]}}$

At test time, batchnorm becomes a linear opeartor

 
#### Layernorm


# Layers

## Convolutional Layer 
**Key Ideas**: 
- **Local Connnectivity**: receptive fields
- **Spatial Arrangement**: Hyperpameters
	- Depth (number of filters)
	- Stride
	- Zero-padding
- **Parameter Sharing**

- Accepts volume of size: $W_1 \times H_1 \times D_1$ 
- Requires 4 Hyperparameters: 
	- $K$: number of filters
	- $F$: their spatial extent
	- $S$: stride
	- $P$: amount of zero padding
- Produces volume size: $W_2 \times H_2 \times D_2$ where:
	- $W_2 = (W_1 - F + 2P)/S + 1$
	- $H_2 = (H_1 - F + 2P)/S + 1$
	- $D_2 = K$
- With parameter sharing, introduces $F \cdot F \cdot D_1$ weights per filter
	- $(F \cdot F \cdot D_1) \cdot K$ total weights
	- $K$ biases
- Ouput volume:
	- d-th depth slice (of size $W_2 \times H_2$) is the result of perming a valid vonvolution of the d-th filter over hte input volume iwth a stride of $S$, and then offseting by d-th bias

#### Dilated Convolution
**Key Idea**: Add new hyperparameter, dilation
#### Grouped Convolutions
Each filter operates on a group of feature maps in the layer / tensor below

Benefits
- Reduces number of parameters and memory reqirements
- Easy Parallelization on several GPUs
- Better representation → less corelated filter responses
#### Deformable Convolutions
Relaxes the fixed geometric structure of standard convolutions
- Standard Convolution:  $y(p_0) = \sum_{p_n \in R} w(p_n) \cdot x(p_0 + p_n)$  
	- R: the set of regular offsets
- Deformable Convolution: $y(p_0) = \sum_{p_n \in R} w(p_n) \cdot x(p_0 + p_n + \Delta p_n)$
	- Learn a set of irregular offsets

Non-parametric approach to geom


## Pooling Layer
**Key idea**: Reduce spatial size
Pooling types
- General (average, L2-Pooling): average
- Max-pooling

Note: falling out of favor- beggingin to see more stride to downsample instead

- Accepts volume of size: $W_1 \times H_1 \times D_1$
- Requires 2 Hyperparameters:
	- $F$: their spatial extent
	- $S$: stride
- Produces volume size $W_2 \times H_2 \times D_2$ where:
	- $W_2 = (W_1 - F )/S + 1$
	- $H_2 = (H_1 - F)/S + 1$
	- $D_2 = D_1$
- Introduces 0 parameters

## FC Layer
**Key Idea**: full connectiosn to all activations in previous layer



## Spatial Transformer Networks
- Parametric Approach to geometric invariance
- Composed of
	- Localization network
	- Grid Sample generator
	- Feature Map Sampler

## **Blocks**

### DenseBlock

DenseBlock
-  Each layer  is connected to every other layer in feedforward fashion

### Squeeze Excitation Block
Squeeze + Excitation Block
- Model channel interdependencies by
	- Squeeze global info in each channel
		- $z_c = F_{sq}(u_c) = \frac{1}{H \times W} \sum_{i =1}^H \sum_{j=1}^W u_c(i,j)$ 
	- Exitation by adaptive recalibration of channel importance
		- $s = F_{ex}(z,W) = \sigma(g(z,W)) =  \sigma(W_2\delta(W_1z))$ 
		- where
	- Re-scaling channels
		- $\textbf{x}_c = F_{scale}(u_c,s_c) = s_c \cdot u_c$



# Architectural Patterns

## Layer Patterns

**Most common**: INPUT → ((CONV → RELU)$*$N → POOL?)$*$M → (FC → RELU)$*$ K → FC
- N≥0, usually N≤3
- M≥0
- K≥0, usually K<3

**Prefer stack of small filter CONV to one large receptive CONV layer**
- For example, one 3x3 > 7x7 convs

**In practice: use whatever works best on ImageNet + finetune to your data**
## Layer Sizing Patterns

**Input Layer** should be divisible by 2 many times

**Conv layers** should use small filters with stride 1

**Pool layers** in charge of downsampling spatial dimensions- most common = 2x2 with stride of 2
- Uncommon to be >3

**Why use stride 1 in CONV?**
- Allows all spatial down-sampling to be done in POOL layers

**Why use padding?**
- Keeps spatial sizes const after CONV + actually improves performance (prevents washing away of info at border)

