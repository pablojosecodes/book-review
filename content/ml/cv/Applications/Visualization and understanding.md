---
title: Visualization and understanding
---
How do we interpret these vision models?
# Visualizing what they have learned

## Filters
Can simply visualize the first layer’s filters by seeing they’re shapes
- Higher than first layer get much more complicated- intractable

## Final Layer Features
Can do cool stuff for a given images’s final feature vectors

For example
- Get L2 neighbors in that feature space
- Visualize the space of feature vectors (using dimensionality reduction)
	- Can even plot the images on an x-y grid and see its underanding
## Activations


# Understanding input pixels

## Important Pixels
Process
- Run images through network- record values of a chosen channel
- Visualize image patches which correspond to maximal activations

## Saliency via occlusion
- Mask part of an image and see how much predicted probabilities change

## Saliency via backprop
Do a forward pass on an image and compute the gradient with respect to image pixels
- Absolute value and max over RGB channels

This will generate a saliency map where white corresponds to impact on the gradient

This can help illuminate biases
- e.g. classifying husky with white snow
## Guided backprop to visualize features
Process
- Pick a single intermediate channel
- Compute gradient of neuron value with respect to image pixels
- Illuminates intermediate features

## Gradient ascent to visualize features
Gradient ascent
- Generate synthetic image which maximally activates a neuron

Process
1. Initialize image to zeros
2. Repeat the following
	1. Forward pass to get current score
	2. Backprop to get gradient of neuron value
	3. Make small update (gradient ascent) to the image

Asecent :  $\underset{I}{\text{arg max}}\ S_c(I) - \lambda||I||^2_2$
- $S_c(I)$: score for class c (before softmax)
- $\lambda||I||^2_2$: simple regularizer
Can do cool stuff with “muti-faceted” visualization
# Adversarial perturbations
General process
1. Pick an artbitrayry image
2. Pick an arbitrary class
3. Modify image to maximize class
4. Repeat until network is fooled

Very subtle changes!
# Style Transfer

## Features Inversion
Given CNN feature vector, get new image whcih
- matches feature vector
- looks natural

Basically
- $x^* = \underset{x \in R^{H\times W \times C}}{\text{arg min}}\ l(\phi(x),\phi_0) + \lambda R(x)$
	- But not $\phi$ todo instead is a similar symbol
	- $l(\phi(x),\phi_0) = ||\phi(x)-\phi_0||^2$

## Deep dream
Instead of synthesizing image to maximize a specific neuron, amlify neuron activations at some layer in the network

Basic process
1. Choose image + layer in CNN
2. Repeat
	1. Compute layer’s activations
	2. Set gradient of layer equal to activation
		1. $I^* = \text{arg max}_l \sum_i f_i(l)^2$ 
	3. Compute gradient on image
	4. Update image
## Texture Synthesis
Goal: patch of texture → bigger image of same texture

Couple of methods

**Nearest neighbor**
- Typical nearest neighbors 
- Generate pixel one at a time in scanline order- form neighborhood of already generated pixels and copy nearest neighbor from input

Neural Texture Synthesis: Gram Matrix
- Each layer of CNN gives 
	- C x H x W tensor of features
	- Equal to: H x W grid of C-dimensoinal vectors
- From outer product of two C-dimensional vectors, get C x C matrix measuring co-occurence
- Average over all HW pairs of vectors, gives
	- **Gram matrix** of shape C x C
Process
1. Pretrain CNN
2. Run input texture forward through CNN, record activations
3. At each layer compute gram matrix
4. Initialize generated image frmo random noise
5. Pass image through CNN, compute gram matrix on each layer
	1. $G_{ij}^l = \sum_k\ F_{ik}^lF_{jk}^l$
	2. Shape is $C_i \times C_i$ 
6. Compute loss
	1. Weighted sum of L2 distance between Gram matrices
7. Backprop to get gradient on image
8. Gradient step on image
9. Go to step 5

## Neural Style Transfer
Feature + gram reconstruction

Basic idea- Content Image + Style Image → Stylized image (ie style transfer)

TODO review

Cons
- Many forward / bacpward passes

Solution: fast style transfer
- Train another neural network to perform style transfer for us

Quick review
- Lots of ways to understand CNN’s representations
- **Activatoins**
	- NN
	- Dimensionality reduction
	- Maximal patches
	- Occlusion
- Gradients
	- Saliency maps
	- Class visualiation
	- Fooling images
	- Feature inversion
- Fun stuff
	- DeepDream- amplify neuron activations at some layer in the network
	- Style transfer- usage of gram matrices