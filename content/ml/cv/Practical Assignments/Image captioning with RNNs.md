---
title: Image captioning with RNNs
---
Let’s use RNNs for image captioning!

We’ll organize into two sections
- RNN Layers: implementing the layers needed for RNNs
	- Forward/backward step
	- Forward/backward pass
	- Word embedding forward/backward pass
	- Temporal Affine Layer
	- Temporal Softmax Layer
	- LSTM Layers
- RNN: Using these layers to implement image captioning
<!---->
# RNN Layers


### Implement a single step forward function for RNN according to these specs

```python
def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """Run the forward pass for a single timestep of a vanilla RNN using a tanh activation function.

    The input data has dimension D, the hidden state has dimension H,
    and the minibatch is of size N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D)
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
```
---
#### Answer
```python
def rnn_step_forward(x, prev_h, Wx, Wh, b):

    next_h, cache = None, None

    # Compute z and pass through tanh. Save cache
    next_h = np.tanh(x @ Wx + prev_h @ Wh + b)
    cache = (next_h, x, prev_h, Wx, Wh)

    return next_h, cache
```

### Implement a single step backward function for RNN according to these specs


```python
def rnn_step_backward(dnext_h, cache):
    """Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None

```

#### Answer

```python
def rnn_step_backward(dnext_h, cache):

    dx, dprev_h, dWx, dWh, db = None, None, None, None, None

    # Retrieve values from cache, compute dz
    next_h, x, prev_h, Wx, Wh = cache
    dz = dnext_h * (1 - np.square(next_h))

    # Compute gradients
    dx = dz @ Wx.T
    dprev_h = dz @ Wh.T
    dWx = x.T @ dz
    dWh = prev_h.T @ dz
    db = dz.sum(axis=0)

    return dx, dprev_h, dWx, dWh, db
```

### Implement RNN forward pass according to these specs and past functions
```python
def rnn_forward(x, h0, Wx, Wh, b):
    """Run a vanilla RNN forward on an entire sequence of data.
    
    We assume an input sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the RNN forward,
    we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D)
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H)
    - cache: Values needed in the backward pass
    """
```
#### Answer
```python
def rnn_forward(x, h0, Wx, Wh, b):
    h, cache = None, None
    
    # Init args
    cache = []
    h = [h0]

    for t in range(x.shape[1]):
        # Run forward pass, retrieve next h and append new cache
        next_h, cache_t = rnn_step_forward(x[:, t], h[t], Wx, Wh, b)
        h.append(next_h)
        cache.append(cache_t)
    
    # Stack over T, excluding h0
    h = np.stack(h[1:], axis=1)

    return h, cache
```

### Implement RNN backward pass according to these specs and past functions
```python
def rnn_backward(dh, cache):
    """Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)
    
    NOTE: 'dh' contains the upstream gradients produced by the 
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None

```
#### Answer
```python
def rnn_backward(dh, cache):

    dx, dh0, dWx, dWh, db = None, None, None, None, None

    # Get the shape values and initialize gradients
    (N, T, H), (D, _) = dh.shape, cache[0][3].shape
    dx = np.empty((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros(H)

    for t in range(T-1, -1, -1):
        # Run backward pass for t^th timestep and update the gradient matrices
        dx_t, dh0, dWx_t, dWh_t, db_t = rnn_step_backward(dh[:, t] + dh0, cache[t])
        dx[:, t] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t

    return dx, dh0, dWx, dWh, db
```


### Implement forward pass for word embedding according to these specs
```python
def word_embedding_forward(x, W):
    """Forward pass for word embeddings.
    
    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """

```

#### Answer
```python
def word_embedding_forward(x, W):
    out, cache = W[x], (x, W)
    
    return out, cache
```


### Implement backward pass for word embedding according to these specs
```python
def word_embedding_backward(dout, cache):
    """Backward pass for word embeddings.
    
    We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D)
    """
```

#### Answer
```python
def word_embedding_backward(dout, cache):
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)
    return dW
```


### Temporal Affine Layer

```python
def temporal_affine_forward(x, w, b):
    """Forward pass for a temporal affine layer.
    
    The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db
```

### Temporal Softmax 
```python
def temporal_softmax_loss(x, y, mask, verbose=False):
    """A temporal version of softmax loss for use in RNNs.
    
    We assume that we are making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores for all vocabulary
    elements at all timesteps, and y gives the indices of the ground-truth element at each timestep.
    We use a cross-entropy loss at each timestep, summing the loss over all timesteps and averaging
    across the minibatch.

    As an additional complication, we may want to ignore the model output at some timesteps, since
    sequences of different length may have been combined into a minibatch and padded with NULL
    tokens. The optional mask argument tells us which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose:
        print("dx_flat: ", dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
```


### LSTM Forward/backward

```python
def lstm_forward(x, h0, Wx, Wh, b):
    """Forward pass for an LSTM over an entire sequence of data.
    
    We assume an input sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running the LSTM forward,
    we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell state is set to zero.
    Also note that the cell state is not returned; it is an internal variable to the LSTM and is not
    accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None

    # Init cell, hidden states and cache list
    c, hs, cache = np.zeros_like(h0), [h0], []

    for t in range(x.shape[1]):
        # Compute hidden + cell state at timestep t, append cache_t to list
        h, c, cache_t = lstm_step_forward(x[:, t], hs[-1], c, Wx, Wh, b)
        hs.append(h)
        cache.append(cache_t)
    
    # Stack along T, excluding h0
    h = np.stack(hs[1:], axis=1)

    return h, cache


def lstm_backward(dh, cache):
    """Backward pass for an LSTM over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None


    # Get the shape values and initialize gradients
    (N, T, H), (D, H4) = dh.shape, cache[0][8].shape
    dx = np.empty((N, T, D))
    dh0 = np.zeros((N, H))
    dc0 = np.zeros((N, H))
    dWx = np.zeros((D, H4))
    dWh = np.zeros((H, H4))
    db = np.zeros(H4)
    
    for t in range(T-1, -1, -1):
        # Run backward pass for t^th timestep and update the gradient matrices
        dx_t, dh0, dc0, dWx_t, dWh_t, db_t = lstm_step_backward(dh0 + dh[:, t], dc0, cache[t])
        dx[:, t] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t


    return dx, dh0, dWx, dWh, db
```
# RNN for Image Captioning

We now have all the necessary layers- so let’s combine them to build an image captioning model.

Fill out the following- the forward and backward pas of the model in the loss function

```python
class CaptioningRNN:
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(
        self,
        word_to_idx,
        input_dim=512,
        wordvec_dim=128,
        hidden_dim=128,
        cell_type="rnn",
        dtype=np.float32,
    ):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {"rnn", "lstm"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # Initialize word vectors
        self.params["W_embed"] = np.random.randn(vocab_size, wordvec_dim)
        self.params["W_embed"] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params["W_proj"] = np.random.randn(input_dim, hidden_dim)
        self.params["W_proj"] /= np.sqrt(input_dim)
        self.params["b_proj"] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        self.params["Wx"] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params["Wx"] /= np.sqrt(wordvec_dim)
        self.params["Wh"] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params["Wh"] /= np.sqrt(hidden_dim)
        self.params["b"] = np.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params["W_vocab"] = np.random.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)
        self.params["b_vocab"] = np.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T + 1) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        - grads: Dictionary of gradients parallel to self.params
        """
```



### Answer
```python
def loss(self, features, captions):
	# Cut captions into two pieces: captions_in has everything but the last word
	# and will be input to the RNN; captions_out has everything but the first
	# word and this is what we will expect the RNN to generate. These are offset
	# by one relative to each other because the RNN should produce word (t+1)
	# after receiving word t. The first element of captions_in will be the START
	# token, and the first element of captions_out will be the first word.
	captions_in = captions[:, :-1]
	captions_out = captions[:, 1:]

	# You'll need this
	mask = captions_out != self._null

	# Weight and bias for the affine transform from image features to initial
	# hidden state
	W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]

	# Word embedding matrix
	W_embed = self.params["W_embed"]

	# Input-to-hidden, hidden-to-hidden, and biases for the RNN
	Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]

	# Weight and bias for the hidden-to-vocab transformation.
	W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

	loss, grads = 0.0, {}


	if self.cell_type == "rnn":
		# If cell type is regular RNN
		recurrent_forward = rnn_forward
		recurrent_backward = rnn_backward
	elif self.cell_type == "lstm":
		# If cell type is long short-term
		recurrent_forward = lstm_forward
		recurrent_backward = lstm_backward

	# Perform forward pass: steps (1) through (4)
	h0, cache_h0 = affine_forward(features, W_proj, b_proj)
	x, cache_x = word_embedding_forward(captions_in, W_embed)
	h, cache_h = recurrent_forward(x, h0, Wx, Wh, b)
	out, cache_out = temporal_affine_forward(h, W_vocab, b_vocab)

	# Compute loss and its partial derivative: step (5)
	loss, dout = temporal_softmax_loss(out, captions_out, mask)

	# Perform backward pass: backpropagate through steps (4) to (1)
	dout, dW_vocab, db_vocab = temporal_affine_backward(dout, cache_out)
	dout, dh0, dWx, dWh, db = recurrent_backward(dout, cache_h)
	dW_embed = word_embedding_backward(dout, cache_x)
	_, dW_proj, db_proj = affine_backward(dh0, cache_h0)

	# Save grads
	grads = {
		"W_proj": dW_proj,
		"b_proj": db_proj,
		"W_embed": dW_embed,
		"Wx": dWx,
		"Wh": dWh,
		"b": db,
		"W_vocab": dW_vocab,
		"b_vocab": db_vocab
	}

	return loss, grads
```

### Create a sample function for the above class according to these specs

```python
def sample(self, features, max_length=30):
	"""
	Run a test-time forward pass for the model, sampling captions for input
	feature vectors.

	At each timestep, we embed the current word, pass it and the previous hidden
	state to the RNN to get the next hidden state, use the hidden state to get
	scores for all vocab words, and choose the word with the highest score as
	the next word. The initial hidden state is computed by applying an affine
	transform to the input image features, and the initial word is the <START>
	token.

	For LSTMs you will also have to keep track of the cell state; in that case
	the initial cell state should be zero.

	Inputs:
	- features: Array of input image features of shape (N, D).
	- max_length: Maximum length T of generated captions.

	Returns:
	- captions: Array of shape (N, max_length) giving sampled captions,
	  where each element is an integer in the range [0, V). The first element
	  of captions should be the first sampled word, not the <START> token.
	"""
	# Unpack parameters
	W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
	W_embed = self.params["W_embed"]
	Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
	W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]
```

#### Answer
Steps
1. Initialize
2. Load
	- Projection weights 
	- Projection biases
	- Embedding weights
	- Hidden state weights
	- Inputs weight
	- Bias weights
	- Weights and biases for the output layer
3. Calculate hidden state (`affine_forward`)
4. Initialize
	1. Hidden state
	2. Starting input
	3. Cell state (for LSTM)
5. For each in length of output:
	1. Get word embedding for current input using `word_embedding`
	2. Generate hidden state (and cell if LSTM) using `step_forward`
	3. Get the output scores using `affine_forward`
	4. Select highest scores
	5. Upate captions

```python
    def sample(self, features, max_length=30):
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # Unpack parameters
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]


        # Initialize the hidden and cell states, input
        h, _ = affine_forward(features, W_proj, b_proj)
        x = np.repeat(self._start, N)
        c = np.zeros_like(h)

        for t in range(max_length):
            # Generate the word embedding of a previous word
            x, _ = word_embedding_forward(x, W_embed)

            if self.cell_type == "rnn":
                # If cell type is regular RNN
                h, _ = rnn_step_forward(x, h, Wx, Wh, b)
            elif self.cell_type == "lstm":
                # If cell type is long short-term memory
                h, c, _ = lstm_step_forward(x, h, c, Wx, Wh, b)

            # Compute the final forward pass for t to get scores
            out, _ = affine_forward(h, W_vocab, b_vocab)
            x = np.argmax(out, axis=1)
            captions[:, t] = x


        return captions
```
