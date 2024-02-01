---
title: Linear Image Classifiers
---
**Image Classification**
Image → probability distribution

Lots of challenges
- Background clutter
- Scale variation
- Deformation 
- Occlusion
- etc.

## **Nearest Neighbor Classifier**
(never used)

**Key idea**: compute pixelwise absolute value difference of image 
- $d(I_1,I-2) = \sum_p|I_1^p - I_2^p|$ (L1 distance as an example)

**“Training”**
- Remember all training data

Prediction: predicts the label of image in training set with nearest score


## **K-Nearest-Neighbor Classifier**
Key idea instead of finding closest image, find k closest images + have them vote on the label of the test image

To apply kNN images in practice
1. Preprocess data
2. Consider using dimensionality reduction technique
3. Split data- train/val
4. Train + evaluate kNN classifier on validation data + across difierent distance types lL1 and L2)
5. If running too long, consider using approximate nearest neighbor library
6. Take note of hyperparameters

## **Linear Classification**
**Core idea**: create an optimization problem by setting up a score function + loss function 

SVM:
- Score function $f(x_i, W, b) = Wx_i + b$
	- Can simplify to $f(x_i, W) = Wx_i$ by appending $x_i$ with one additional dimensions



Loss function

**Multiclass support vector machine leoss**

For i-th example we have pixels and labels, and then score function gives vector of class scores
- Loss would be $L_i = \sum_{j \neq y_i}\text{max}(0,s_j-s_{y_j}+\Delta)$ or $L_i = \sum_{j \neq y_i} \text{max}(0, w^T_jx_i =- w^T_{y_i}x_i + \Delta)$ 
- Basically, sum of max(0, difference between confidence in this prediction - prediction in actual class) over all the incorrect classes 

**Hinge loss**: typically what we call the threshold at zero “max$(0,-)$” 
- Squared hinge loss:  “$\text{max}(0,-)^2$”


## **Softmax**
Core idea: new loss function

Loss function becomes
- $L_i = -\text{log}(\text{softmax}(\text{prediction outputs}))$ 


**Information Theory View**
- Cross entropy: $H(p,q) = - \sum_x p(x) \text{log}\ q(x)$
- Thus, softmax minimizes cross entropy between estiamted class probabilities and hte “true”distirbution

## SVM vs softmax

Softmax
- Provides probabilities for each class

SVM
- Computes uncalibrated + not easy to interpret scores for all classes
- More local

Usually similar performance
