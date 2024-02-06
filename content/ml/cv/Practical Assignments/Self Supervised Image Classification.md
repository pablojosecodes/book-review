---
title: Self Supervised Image Classification
---
Self supervised learning

What makes a good representation?
- A good representation vector captures the important features of the image as it relates to the rest of the dataset

Contrastive Learning: SimCLR
- New architecture which uses **contrastive learning**

Given an image x, SimCLR uses two data augmentation schemes **t** and **t’** to generate positive parif of images.

# Pretrained weights
Available at http://downloads.cs.stanford.edu/downloads/cs231n/pretrained_simclr_model.pth


## Implement `compute_train_transform()` to apply random transformations, as well as __get__item
The random transformations
1. Randomly resize and crop to 32x32.
2. Horizontally flip the image with probability 0.5
3. With a probability of 0.8, apply color jitter
4. With a probability of 0.2, convert the image to grayscale 

```python
def compute_train_transform(seed=123456):
    """
    This function returns a composition of data augmentations to a single training image.
    Complete the following lines. Hint: look at available functions in torchvision.transforms
    """
    random.seed(seed)
    torch.random.manual_seed(seed)
    
    # Transformation that applies color jitter with brightness=0.4, contrast=0.4, saturation=0.4, and hue=0.1
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
    
    train_transform = transforms.Compose([

		# CODE GOES HERE
        
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return train_transform


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        x_i = None
        x_j = None

        if self.transform is not None:

		# CODE GOES HERE

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x_i, x_j, target
```


### Answer
```python
def compute_train_transform(seed=123456):

    random.seed(seed)
    torch.random.manual_seed(seed)
    
    # Transformation that applies color jitter with brightness=0.4, contrast=0.4, saturation=0.4, and hue=0.1
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
    
    train_transform = transforms.Compose([
        # Step 1: Randomly resize and crop to 32x32.
        transforms.RandomResizedCrop(32),
        # Step 2: Horizontally flip the image with probability 0.5
        transforms.RandomHorizontalFlip(p=0.5),
        # Step 3: With a probability of 0.8, apply color jitter (you can use "color_jitter" defined above.
        transforms.RandomApply([color_jitter], p=0.8),
        # Step 4: With a probability of 0.2, convert the image to grayscale
        transforms.RandomGrayscale(p=0.2),

        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return train_transform


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        x_i = None
        x_j = None

        if self.transform is not None:
            x_i = self.transform(img)
            x_j = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x_i, x_j, target
```


## Understand SimCLR Mode
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
```

# SimCLR: Contrastive Loss

## Implement `sim`

```python
def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.
    
    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = None
	# CODE GOES HERE
    
    return norm_dot_product
```

### Answer
```python
def sim(z_i, z_j):
    
    norm_dot_product = (z_i/torch.linalg.norm(z_i)) @ (z_j/torch.linalg.norm(z_j))
    
    
    return norm_dot_product
```
## Implement `simclr_loss_naive`
```python

def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).
    
    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair. 
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.
    
    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
    """
    N = out_left.shape[0]  # total number of training examples
    
     # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]
        
	# CODE GOES HERE
    
    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2*N)
    return total_loss
```

### Answer
```python

def simclr_loss_naive(out_left, out_right, tau):

    N = out_left.shape[0]  # total number of training examples
    
     # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]

        # Create lists of non-exponentiated unnormalized similarity scores
        sims_k = torch.tensor([sim(z_k, z) for z in out[np.arange(2*N)!=k]])
        sims_k_N = torch.tensor([sim(z_k_N, z) for z in out[np.arange(2*N)!=k+N]])

        # Compute l(k, k+N) and l(k+N, k), given the lists of similarity scores
        l1 = -((sim(z_k, z_k_N) / tau).exp() / (sims_k / tau).exp().sum()).log()
        l2 = -((sim(z_k_N, z_k) / tau).exp() / (sims_k_N / tau).exp().sum()).log()

        # Update the total loss
        total_loss += l1 + l2
    
    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2*N)
    return total_loss
```
## Implement `sim_positive_pairs`
```python
def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.
    
    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    pos_pairs = None
    
	# CODE GOES HERE

    return pos_pairs
```

### Answer 
```python
def sim_positive_pairs(out_left, out_right):

    norm_left = out_left / torch.linalg.norm(out_left, dim=1, keepdim=True)
    norm_right = out_right / torch.linalg.norm(out_right, dim=1, keepdim=True)

    # Compute the diagonal dot product directly by multiplying and summing
    pos_pairs = (norm_left * norm_right).sum(dim=1, keepdim=True)


    

    return pos_pairs
```

## Implement `compute_sim_matrix`
```python

def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.
    
    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """
    sim_matrix = None
    
	# CODE GOES HERE
	
    return sim_matrix
```

### Answer
```python

def compute_sim_matrix(out):

    norm_out = out / torch.linalg.norm(out, dim=1, keepdim=True)
    sim_matrix = norm_out @ norm_out.T

    return sim_matrix
```

## Implement `simclr_loss_vectorized`
```python
def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.
    
    Inputs and output are the same as in simclr_loss_naive.
    """
    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    
    # CODE GOES HERE
    
    return loss
```

### Answer
```python
def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):

    N = out_left.shape[0]
    
    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]
    
    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]
    

    # Step 1: Use sim_matrix to compute the denominator value for all augmented samples.
    # Hint: Compute e^{sim / tau} and store into exponential, which should have shape 2N x 2N.
    exponential = (sim_matrix / tau).exp().to(device)
    
    # This binary mask zeros out terms where k=i.
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()
    
    # We apply the binary mask.
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]
    
    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    denom = exponential.sum(dim=1)

    # Step 2: Compute similarity between positive pairs.
    # You can do this in two ways: 
    # Option 1: Extract the corresponding indices from sim_matrix. 
    # Option 2: Use sim_positive_pairs().

    sim_pairs = sim_matrix[range(2*N), [*range(N, 2*N), *range(0, N)]]

    
    # Step 3: Compute the numerator value for all augmented samples.
    numerator = (sim_pairs / tau).exp().to(device)
    
    # Step 4: Now that you have the numerator and denominator for all augmented samples, compute the total loss.
    loss = -(numerator / denom).log().mean()
    
    
    return loss
```



# Implement train function
```python
def train(model, data_loader, train_optimizer, epoch, epochs, batch_size=32, temperature=0.5, device='cuda'):
    """Trains the model defined in ./model.py with one epoch.
    
    Inputs:
    - model: Model class object as defined in ./model.py.
    - data_loader: torch.utils.data.DataLoader object; loads in training data. You can assume the loaded data has been augmented.
    - train_optimizer: torch.optim.Optimizer object; applies an optimizer to training.
    - epoch: integer; current epoch number.
    - epochs: integer; total number of epochs.
    - batch_size: Number of training samples per batch.
    - temperature: float; temperature (tau) parameter used in simclr_loss_vectorized.
    - device: the device name to define torch tensors.

    Returns:
    - The average loss.
    """
    model.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for data_pair in train_bar:
        x_i, x_j, target = data_pair
        x_i, x_j = x_i.to(device), x_j.to(device)
        
        out_left, out_right, loss = None, None, None

		# CODE GOES HERE
		
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num
```

### Answer
```python
def train(model, data_loader, train_optimizer, epoch, epochs, batch_size=32, temperature=0.5, device='cuda'):

    model.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for data_pair in train_bar:
        x_i, x_j, target = data_pair
        x_i, x_j = x_i.to(device), x_j.to(device)
        
        out_left, out_right, loss = None, None, None

        (_, out_left), (_, out_right) = model(x_i), model(x_j)
        loss = simclr_loss_vectorized(out_left, out_right, temperature, device)
        
        
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num
```
