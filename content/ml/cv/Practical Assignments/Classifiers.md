---
title: Classifiers
---


# K Nearest Neighbors

Compute distance 

```python
def compute_distance(self, X):
	"""
	Compute the distance between each test point in X and each training point
	in self.X_train using a nested loop over both the training data and the
	test data.

	Inputs:
	- X: A numpy array of shape (num_test, D) containing test data.

	Returns:
	- dists: A numpy array of shape (num_test, num_train) where dists[i, j]
	  is the Euclidean distance between the ith test point and the jth training
	  point.
	"""

```

Answer
```python
def compute_distance(self, X):
	num_test = X.shape[0]
	num_train = self.X_train.shape[0]
	dists = np.zeros((num_test, num_train))
	dists = np.sqrt(
          -2 * (X @ self.X_train.T) +
	np.power(X, 2).sum(axis=1, keepdims=True) +
	np.power(self.X_train, 2).sum(axis=1, keepdims=True).T
	)
	return dists
```

Implement cross validation with the following parameters
```python
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
```
Answer

```python
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []


X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies = {}

for k in k_choices:
    k_to_accuracies[k] = []

    for i in range(num_folds):
        # Create num_folds-1 of the data and labels as the training samples 
        X_train_temp = np.concatenate(np.compress(np.arange(num_folds) != i, X_train_folds, axis=0))
        y_train_temp = np.concatenate(np.compress(np.arange(num_folds) != i, y_train_folds, axis=0))

        # Train the classifier based on the training data
        classifier.train(X_train_temp, y_train_temp)

        # Predict using the remaining fold representing validation data
        y_pred_temp = classifier.predict(X_train_folds[i], k=k, num_loops=0)

        # Compute the accuracy of the predicted label
        num_correct = np.sum(y_pred_temp == y_train_folds[i])
        k_to_accuracies[k].append(num_correct / len(y_pred_temp))



# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))

```


# Support Vector Machine Classsifier


Calculate the SVM gradient and loss according to specs
```python
def svm_loss(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
```

Answer
```python
def svm_loss(W, X, y, reg):

    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    N = len(y)     # number of samples
    Y_hat = X @ W  # raw scores matrix

    y_hat_true = Y_hat[range(N), y][:, np.newaxis]    # scores for true labels
    margins = np.maximum(0, Y_hat - y_hat_true + 1)   # margin for each score
    loss = margins.sum() / N - 1 + reg * np.sum(W**2) # regularized loss

    dW = (margins > 0).astype(int)    # initial gradient with respect to Y_hat
    dW[range(N), y] -= dW.sum(axis=1) # update gradient to include correct labels
    dW = X.T @ dW / N + 2 * reg * W   # gradient with respect to W

    return loss, dW
```

Given the above loss function, calculate training with sgd according to the  specs
```python
def train(
	self,
	X,
	y,
	learning_rate=1e-3,
	reg=1e-5,
	num_iters=100,
	batch_size=200,
	verbose=False,
):
	"""
	Train this linear classifier using stochastic gradient descent.

	Inputs:
	- X: A numpy array of shape (N, D) containing training data; there are N
	  training samples each of dimension D.
	- y: A numpy array of shape (N,) containing training labels; y[i] = c
	  means that X[i] has label 0 <= c < C for C classes.
	- learning_rate: (float) learning rate for optimization.
	- reg: (float) regularization strength.
	- num_iters: (integer) number of steps to take when optimizing
	- batch_size: (integer) number of training examples to use at each step.
	- verbose: (boolean) If true, print progress during optimization.

	Outputs:
	A list containing the value of the loss function at each training iteration.
	"""
```

Answer
```python
    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )  # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]


            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

```


# Softmax Classifer


- Multilayer Neural Network


Convolutional architectures
- Multilyaer Fuly Connected Networks
	- 

Specific Applications / Advanced Architctures
- 