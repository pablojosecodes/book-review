---
title: Generative Models
---

Remmeber the calsses
- Supervised vs unsupervised
- Supervised: learn $x → y$- given (x, y)
	- Semantic segmentation
	- Object detection
	- Image captioning
	- Classification
- Unsupervised: learn some underlying hidden *structure* of the data- given just data, no labels
	- Clustering
	- Dimensionality reduction
	- Feature learning
	- Density estimation

Discriminative vs Generative (for data x and label y)
- Discriminative: learn $p(y|x)$
	- Can’t handle unreasonable inputs- gives label distribution for all images
	- **Functoin**: assigment label to data (feature learning with labels)
- Generative: learn $p(x)$ 
	- Requires deep understanding of images
	- Can “reject” unreasonable inputs
	- **Functions**: detect outliers, feature learning without labels, sample from to generate new data
- Conditional genereative model- learn $p(x|y)$
	- Assign labels while rejecting outliers
	- Generate new dat conditioned on input labels

Remember Bayes’ Rule
- $p(x|y) = \frac{p(y|x)}{p(y)}p(x)$
- Well, we can build if we learn all of these!
	- And $p(y)$ is prior over labels

**Taxonomy of generative models**
- **Explicit** density: can compute $p(x)$
	- **Tractable** density- can compute $p(x)$
	- **Approximate** density- approximation to $p(x)$
		- **Variational**
		- **Markov Chain**
- **Implicit** density- does nt explicitly compute $p(x)$ but can sample from
	- **Markov Chain**
	- **Direct**



# Autoregressive models

Tractable density model

Gaol: write down $p(x) = f(x, W)$
$W*  = \underset{W}{\text{arg max}}\prod_ip(x_{(i)}) =  \underset{W}{\text{arg max}} \sum_i \text{log} p(x^{(i)}) = \underset{W}{\text{arg max}} \sum_i \text{log} f(x^{(i)},W)$ 
$P(x) = \prod_{t=1}^T p(x_t| x_1, \ldots, x_{t-1})$
- proability of next subpart givcen all previous subparts

## PixelRNN
Generates pixes one t a time

Compute hidden state for each pixel which depends on hidden states / RGB alues from left + above (LSTM recurrence)

$h_{x,y} = f(h_{x-1, y}, h_{x,y-1},W)$

Each pixel- preict red, then bue, then green- softmax over $[0,1,\ldots, 255]$

Problem: Slow (N x N image requires 2N - 1 sequential steps)

## PixelCNN
- still start from corner- but dependency is modeled using a CNN

Training: still maximize the likelihood of training images $P(x) = \prod_{t=1}^T p(x_t| x_1, \ldots, x_{t-1})$ 

Still generate starting from corner

Faster, but still sequential generation

# Variational Autoencoders

## Regular Autoencoders
- Unsupervised method for learning feature vectors, by doing encoding then decoding, without label s just raw data
- Basic loss function of MSE- L2 distance between input + reconstructed data

After training, throw away decoder- use encoder for downstream task

These autoencoders learn latent features

## Variational Autoencoders
Probabilistic spin on rergualr autoencoders
1. learn latent features
2. Sample from model ot generate new data

Assume training data generated from latent representation
- Intuition: x is an image, z is latent factors used to generate x

Sampling new data
- Sample from conditional $p_{\theta*}(x | z_{(i)})$
- Sample z from prior $p_{\theta*}(z)$

Assume a simple prior $p(z)$- e.g. Gaussa

Represent $p(x|z)$ with neural network (similar to decoder from auto-encoder)

Decoder must be **probabilistic**
- But how?

Encoder network
- $q_\phi(z | x)= N(\mu_{z|x}, \sum_{z|x})$ 
- Input: x (image, flattened to vector)

Decoder network
- $p_\theta(x|z) =  N(\mu_{z|x}, \sum_{z|x})$ 
- Input: z (vector)

Note that bosth use the diagonal guassian trick

Training goal:
- Maximizing variational lower bound $E_{z \sim q_\phi(z|x)}[\text{log}\ p_\theta(x|z)]-D_{KL}(q_\phi(z|x), p(z))$

How to train though?
1. Run input through **encoder** to get distribution over latent codes
2. Output should match the prior $p(z)$
	1. Second term
	2. Basically we want the output to match the prior we have chosen
3. Sample code $z$ from encoder output $(z \sim q_\phi(z|x))$ 
4. Run code through **decoder** → distribution over data samples
	1. We want to maximize the likelihood of the data of x under predicted distributino of the decoder when we feed in a sample z
	2. Data reconstruction term
5. Original input data should be under distribution output from step 4
6. Can sample a reconstruction from 4

basically, we’re trying to put some kind of limit (through KL divergence) on the kind of latent variables we’re trying to predict, while jointy training with the decoder network that reconstructs the latent variables into image

## once trained

Generating new data
1. Sample $z$ from prior $p(z)$
2. Run $z$ through decoder to get distribution over data $x$
3. sample from distribution in step 2 to generate data

Editing images after training

Since we enforce diagonal prior on distribution of $p(z)$, the dimensions are independent, so each latent variable should encode something different
1. Run input data through encoder to get distirbution over latent codes
2. Sample code $z$ from encoder output
3. Modify some dimensions of sampled code
4. Run modified z through decoder to get distribution over data sample
5. Sample new data from step 4

Summary of variational autoencoders
- Probabilisitc spin on traditional autoencoders to allow for data generation
- Functions: define intractable density → derive + optimize variational lower bound
- Pro
	- Principled approach
	- Allows inference of $q(z | x)$- can be useful for other tasks
- Cons
	- Maximizes lower bound of likelihood- not super good evaluation
	- Blurrier samples than other methods


Generative models summary
- Autoregressive: directly maximize likelihood of training data
	- $p_\theta(x) = \prod_i p_\theta(x_i|x_1,\ldots,x_{i-1})$ 
- Variational autoencoders: introduce latent $z$- maximize a lower bound
	- $p_\theta(x) = \int_z p_\theta(x|z)p(z)dz \geq E_{z \sim q_\phi(z|x)}[log p_\theta(x|z)] - D_{KL}(q_\phi(z|x),p(z))$ 
- GANs: give up on modeling $p(x)$, but be able to sample from $p(x)$

# GANs

Assume that we have $x_i$ drawn from $p_{\text{data}}(x)$- we want to sample from $p_{\text{data}}$ 


Idea: introduce latent variable $z$ with simple prior $p(z)$

Sample $z \sim p(z)$ and pass to generator → x = g(z)

Then, x is sample from the generator distribution.
- Goal: We want $p_g = p_{\text{data}}$

How?
- Train G to convert z into fake data x sampled from $p_g$ by fooling discriminator
- Goal: $p_g$ converges to $p_\text{data}$ 

Train discriminator to classify data as real or fake

Called the minimax game: $\underset{G}{\text{min}}\ \underset{D}{\text{max}} (E_{x \sim p_\text{data}}[\text{log}D(x)]+E_{z \sim p(z)}[\text{log}(1-D(G(z)))])$ 

Above equation just shows
- Discriminator is trying to maximize 
	- The probability that real data is classified as 1 
	- The probability that fake data is classified as 0
- Generator is trying to minimize
	- probability that fake data is classified as 1

Train G and D with alternative gradient updates


But how do we look at loss??? No overall loss or training curves
- Pretty challenging to train

Beginning of training
- Generator sucks- vanishing gradients
- Solution: train G to maximize $-log(D(G(z)))$ instead of minimizing $log(1 - D(G(z)))$ 

## Some GAN Architectures

Can interpolate between points in latent space $z$

Lots of stuff more recently probably