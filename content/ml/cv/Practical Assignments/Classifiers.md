---
title: Classifiers
---


# K Nearest Neighbors

#### Create a KNN distance computation according to these specs 

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

#### Answer
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

####  Implement cross validation with the following parameters
```python
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
```

#### Answer
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

####  Calculate the SVM gradient and loss according to specs
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

#### Answer
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

####  Given the above loss function, calculate training with sgd according to these specs
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

#### Answer
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

#### Implement gradient and loss for softmax classifier given these specs
```python
def softmax_loss(W, X, y, reg):
    """
    Softmax loss function

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

Remember that the softmax function is essentially SVM with a different loss function and no hinge loss

This means that you
1. Get the raw outputs (logits)
2. Normalize (subtract by max)
3. Turn into exponents
4. Get the softmax  values (softmax equation)

Then, to get loss
5. Get the loss from the softmax (negative log) for the index of what would be correct prediction
6. Regularize + average

To get gradient
7. Update the correct classification (subtract 1)
8. Calculate the gradient

####  **Answer**
```python
def softmax_loss(W, X, y, reg):
	loss = 0
	dW = None
	N = X.shape[0]

	Y_hat = X @ W  # 1) raw scores matrix
	P = Y_hat - Y_hat.max() # 2) normalize
	P = np.exp(P)          # 3) Get exponent values 
	P /= P.sum(axis=1, keepdims=True)    # 4) Get softmax values
	loss = -np.log(P[range(N), y]).sum() # 5) Get loss from softmax
	loss = loss / N + reg * np.sum(W**2) # 6) Avg. + regularize 
	P[range(N), y] -= 1                  # 7) Update classification
	dW = X.T @ P / N + 2 * reg * W       # 8) Get gradient
	
	return loss, dW
```




# Fully Connected Neural Network


#### Create an `affine_forward` function according to these specs
```python
def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
```

#### Answer
```python
def affine_forward(x, w, b):
	out = None

	x_reshaped = x.reshape(x.shape[0], -1)
	out = x_reshaped @ w + b

	cache = (x, w, b)
	return out, cache
```

#### Create an `affine_backward` function according to these specs
```python
def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
```

#### Answer
```python
def affine_backward(dout, cache):
	x, w, b = cache

    x_reshaped = x.reshape(x.shape[0], -1)
    dx = (dout @ w.T).reshape(x.shape[0], *x.shape[1:])
    dw = x_reshaped.T @ dout
    db = dout.sum(axis=0)

	return (dx, dw, db)
```

#### Create a `forward pass for ReLU` activation function according to these specs

```python
def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
```

#### Answer
```python
def relu_forward(x):
	cache = x
	out = np.maximum(x, 0)
	return (out, cache)
```

#### Create a `backward pass for ReLU` activation function according to these specs
```python
def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
```
#### Answer
```python
def relu_backward(dout, cache):
    dx, x = None, cache
    dx = dout * (x > 0)
    return dx
```


#### Implement `svm_loss` according to these specs
```python
def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """

```

#### Answer
```python
def svm_loss(x, y):
    loss, dx = None, None

    N = len(y)                                # number of samples
    x_true = x[range(N), y][:, None]          # scores for true labels
    margins = np.maximum(0, x - x_true + 1)   # margin for each score
    loss = margins.sum() / N - 1
    dx = (margins > 0).astype(float) / N
    dx[range(N), y] -= dx.sum(axis=1)

    return loss, dx
```



#### Implement `softmax_loss` according to these specs
```python
def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
```

#### Answer
```python
def softmax_loss(x, y):

    loss, dx = None, None

    N = len(y) # number of samples

    P = np.exp(x - x.max(axis=1, keepdims=True)) # numerically stable exponents
    P /= P.sum(axis=1, keepdims=True)            # row-wise probabilities (softmax)

    loss = -np.log(P[range(N), y]).sum() / N # sum cross entropies as loss

    P[range(N), y] -= 1
    dx = P / N

    return loss, dx
```


#### Build out a full Two Layer Neural Network according to these specs
```python
class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """

```

#### Answer
```python
class TwoLayerNet(object):

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):

        self.params = {}
        self.reg = reg

        self.params = {
          'W1': np.random.randn(input_dim, hidden_dim) * weight_scale,
          'b1': np.zeros(hidden_dim),
          'W2': np.random.randn(hidden_dim, num_classes) * weight_scale,
          'b2': np.zeros(num_classes)
        }


    def loss(self, X, y=None):

        scores = None

        W1, b1, W2, b2 = self.params.values()

        out1, cache1 = affine_forward(X, W1, b1)
        out2, cache2 = relu_forward(out1)
        scores, cache3 = affine_forward(out2, W2, b2)


        if y is None:
            return scores

        loss, grads = 0, {}

        
        loss, dloss = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2))

        dout3, dW2, db2 = affine_backward(dloss, cache3)
        dout2 = relu_backward(dout3, cache2)
        dout1, dW1, db1 = affine_backward(dout2, cache1)

        dW1 += self.reg * W1
        dW2 += self.reg * W2

        grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

        return loss, grads
```
