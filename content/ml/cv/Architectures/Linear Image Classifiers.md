---
title: Linear Image Classifiers
---
**Image Classification** is the task of assigning an Image a probability distribution (on a set of classes)

There are lots of practical matters which make this a difficult problem
- Background clutter
- Scale variation
- Deformation 
- Occlusion
- etc.

We’ll talk about the following classifiers
- Nearest Neighbor
- Support Vector Machines
- Softmax

## **Nearest Neighbor Classifier**

**Key idea**: compute pixelwise absolute value difference of image 
- $d(I_1,I_2) = \sum_p|I_1^p - I_2^p|$ (L1 distance)
- $d(I_1,I_2) = \sqrt{\sum_p (I_1^p - I_2^p)^2}$ (L2 distance)

**Training**
- Quite literally just load all the training data

**Predicting**: 
- Predict label of image in training set with the neasrest score

## **K-Nearest-Neighbor Classifier**
**Key idea**: find $k$ closest images and have them vote on the label of the test image

Apply K-NN to images in practice
1. Preprocess data
2. Consider using dimensionality reduction technique
3. Split data into training and and evaluation sets
4. Train and evaluate the kNN classifier on validation data + across difierent distance types (e.g. L1 and L2)
5. If your code is running too long, consider using an ANN library
6. Take note of your hyperparameters!

## **Linear Classification**
**Core idea**: create an optimization problem by setting up a score function + loss function 

Support vector machines:
- Score function $f(x_i, W, b) = Wx_i + b$
	- Can simplify to $f(x_i, W) = Wx_i$ by appending $x_i$ with one additional dimensions



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
