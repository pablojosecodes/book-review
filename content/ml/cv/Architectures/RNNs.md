---
title: RNNs
---

Basics covered in ML section


## Image Captioning
Goal: input image, feed to CNN to get features, get caption from features using RNN

- Transfer learning: take CNN trained on ImageNet + chop po f last layer
- Feed `<START> `token as RNN first input and feed CNN output as extra input to each hidden layer
	- $h = \text{tanh}(W_{xh}* x + W_{hh} * h + W_{ih} * v)$
- Stop when `<END>` token emitted