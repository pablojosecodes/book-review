---
title: Semantic Segmentation
---

 
 **Goal**: Attach to each pixel in an image a label from a set of predefined classes
### Sliding Window Approach 
(Naive approach):  Classify one pixel per run
- Center sliding window onto a pixel and push through net to establish its labels.
- Sucks because
	- Need a lot of data
	- Inefficient
	- Can’t use large neighborhoods
	- No parameter reuse

### Fully Convolutional Neural Net (FCNN) (better approach)
- Behaves as a huge filter (input size is arbitrary, and output size depends on input)
- Why is it better? 
	- Provides a featmap for each class
	- Efficient evluation. end to end training
	- Res use of shared parameters
	- Less parameters
- Transposed convolution: 
- Recover inital image resolution with the transposed convolution
- What’s the problem?
	- Spectrum of deep features
		- Combine where with what
	- Solution: add skip connections from finer convolutional layers

- Super computational heavy
- We like to reduce feature spatial size

Now instead, down and upsample
- Downsample: pooling, strided convolution
- Upsmpling: unpooling
	- Unpooling
		- Nearest neighbor
		- Bed of nails
		- Max unpooling (remember which element on grid was the max- use those positions for bed of nails)
	- Learnable upsampling (with transposed convolutions,  strided)
		- Learn filter which takes weights from input to upsample

Loss 
- cross entropy basically


### AutoEncoder architectures
#### Encoder-Decoder (alternative approach)
- Encoder
	- VGG16-baed (13 conv layers)
	- Conv layers 3x3, stride 1 + batch norm + ReLU
	- Max Pooling 2x2, stride 2
	- Stored max pool indices (for later upsampling
- Decoder
	- Unsampled Sparse feature map
	- Transposed convolutions decoder filter bank
	- Batch Norm + ReLU
- Unpooling
- classification
	- Multiclass softmax trainable classifier (each pixel is a soft max)
	- class requency balancing
#### Encoder-decoder + Skip connections (eclectic approach)
- Stacked hourglass
