# Definitions:
* PyTorch is a popular deep learning framework with syntax similar to NumPy.

* In PyTorch, a Tensor functions like a NumPy array that runs on a CPU or GPU, with support for automatic differentiation, essential for backpropagation.

* TorchText, TorchVision, and TorchAudio are Python packages that expand PyTorch's functionality to text, image, and audio data.

* A neural network is made up of layers of neurons. Each layer's neurons receive input values and process them through weights and biases. A neuron's output is a weighted sum of inputs plus bias, passed to the next layer's neurons, continuing until the network's final layer is reached.

* An activation function transforms a neuron's output, introducing non-linearity into the model's calculations.

* Backpropagation trains neural networks by iteratively adjusting neurons' weights and biases to reduce error.

* Saturation happens when a neuron's output reaches an extreme beyond which it can’t change, potentially hindering learning. Activation functions like ReLU can help prevent this.

* The loss function measures the difference between the model's predicted and actual outputs.

* The optimizer adjusts neural network parameters to minimize the loss function during training.

* The learning rate determines the optimizer’s step size. A low learning rate can slow down training, while a high one may lead to unstable convergence.

* Momentum controls the inertia of an optimizer. Low momentum can cause the optimizer to get stuck in local minima, while high momentum might overshoot the solution.

* Transfer learning uses a model trained on one task as a starting point for another task, speeding up training.

* Fine-tuning involves freezing early layers and only training later layers, focusing on new output requirements.

* Accuracy is a metric that evaluates how well a model fits a dataset, measuring the ratio of correct predictions to total data points.




## Importing PyTorch
1. Import the core PyTorch package
``` 
import torch
```

2. Import neural network functionality
```
from torch import nn
```

3. Import functional programming tools
```
import torch.nn.functional as F
```

4. Import optimization functions
```
import torch.optim as optim
```

5. Import dataset tools
```
from torch.utils.data import TensorDataset, DataLoader
```

6. Import evaluation metrics
```
import torchmetrics
```

## Working with Tensors

1. Create a tensor from a list using tensor()
```
tnsr = torch.tensor([1, 3, 6, 10])
```

2. Get the data type of tensor elements with .dtype
```
tnsr.dtype  # Returns torch.int64
```

3. Get the dimensions of the tensor with .size()
```
tnsr.shape  # Returns torch.Size([4])
```

4. Check the device location of the tensor with .device
```
tnsr.device  # Returns either CPU or GPU
```

5. Create a tensor of zeros using zeros()
```
tnsr_zrs = torch.zeros(2, 3)
```

6. Create a random tensor with rand()
```
tnsr_rndm = torch.rand(size=(3, 4))  # Tensor has 3 rows, 4 columns
```


## Datasets and Dataloaders

1. Create a dataset from a pandas DataFrame using TensorDataset()
```
X = df[feature_columns].values
y = df[target_column].values
dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())
```

2. Load the data in batches with DataLoader()
```
dataloader = DataLoader(dataset, batch_size=n, shuffle=True)
```
## Preprocessing
1. One-hot encode categorical variables with one_hot()
```
F.one_hot(torch.tensor([0, 1, 2]), num_classes=3)  # Returns a tensor of 0s and 1s
```
## Sequential Model Architecture
1. Create a linear layer with m inputs and n outputs using Linear()
```
lnr = nn.Linear(m, n)
```
2. Get the weight of the layer with .weight
```
lnr.weight
```

3. Get the bias of the layer with .bias
```
lnr.bias
```

4. Create a sigmoid activation layer for binary classification with Sigmoid()
```
nn.Sigmoid()
```

5. Create a softmax activation layer for multi-class classification with Softmax()
```
nn.Softmax(dim=-1)
```
6. Create a ReLU activation layer to prevent saturation with ReLU()
```
nn.ReLU()
```

7. Create a LeakyReLU layer to prevent saturation with LeakyReLU()
```
nn.LeakyReLU(negative_slope=0.05)
```

8. Create a dropout layer to regularize and reduce overfitting with Dropout()
```
nn.Dropout(p=0.5)
```

9. Define a sequential model with layers
```
model = nn.Sequential(
    nn.Linear(n_features, i),
    nn.LeakyReLU(),
    nn.Linear(i, j),  # Input size must match the previous layer's output size
    nn.Linear(j, n_classes),
    nn.Softmax(dim=-1)  # The activation layer should be the last layer
)
```





## Fitting a Model and Calculating Loss
1. Make Predictions
* Fit a model to input data:
```
prediction = model(input_data).double()
```
2. Define Target Values
* Get target values:
```
actual = torch.tensor(target_values).double()
```
3. Calculate Losses

* Mean-Squared Error Loss (for regression):
```
mse_loss = nn.MSELoss()(prediction, actual)  # Returns tensor(x)
```
* L1 Loss (for robust regression):
```
l1_loss = nn.SmoothL1Loss()(prediction, actual)  # Returns tensor(x)
```
* Binary Cross-Entropy Loss (for binary classification):
```
bce_loss = nn.BCELoss()(prediction, actual)  # Returns tensor(x)
```
* Cross-Entropy Loss (for multi-class classification):
```
ce_loss = nn.CrossEntropyLoss()(prediction, actual)  # Returns tensor(x)
```

4. Backpropagation
* Calculate gradients via backpropagation:
```
loss.backward()
```

## Working with Optimizers
1. Define Optimizer
* Create a stochastic gradient descent (SGD) optimizer, setting learning rate and momentum:

```
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
```

2. Update Model Parameters
* Update neuron parameters:
```
optimizer.step()
```


## The Training Loop
1. Set Model to Training Mode

```
model.train()
```

2. Define Loss Criterion and Optimizer
```
loss_criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
```

3. Loop Through Data Chunks in Training Set
``` for data in dataloader:
    # Reset gradients
    optimizer.zero_grad()

    # Separate features and targets
    features, targets = data

    # Forward Pass: Fit model to data
    predictions = model(data)

    # Calculate Loss
    loss = loss_criterion(predictions, targets)

    # Backpropagation: Compute gradients
    loss.backward()

    # Update Model Parameters
    optimizer.step()
```
    
## The Evaluation Loop
1. Set Model to Evaluation Mode
```
model.eval()
```

2. Define Accuracy Metric
```
metric = torchmetrics.Accuracy(task="multiclass", num_classes=3)
```

3. Loop Through Data Chunks in Validation Set
```
for i, data in enumerate(dataloader, 0):
    # Separate features and targets
    features, targets = data

    # Forward Pass: Fit model to data
    predictions = model(data)

    # Calculate Accuracy Over Batch
    accuracy = metric(predictions.argmax(dim=1), targets)


    # Calculate Overall Accuracy for Validation Data
    accuracy = metric.compute()
    print(f"Accuracy on all data: {accuracy}")
    
    #Reset Metric for Next Dataset
    metric.reset()
```


# Transfer Learning and Fine-Tuning
1. Save a Model Layer
```
torch.save(layer, 'layer.pth')
```

2. Load a Saved Model Layer
```
new_layer = torch.load('layer.pth')
```

3. Freeze Layer Weights for Fine-Tuning
```
for name, param in model.named_parameters():
    if name == "0.weight":
        param.requires_grad = False
```
