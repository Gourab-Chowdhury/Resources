Definitions:
PyTorch is a popular deep learning framework with syntax similar to NumPy.

In PyTorch, a Tensor functions like a NumPy array that runs on a CPU or GPU, with support for automatic differentiation, essential for backpropagation.

TorchText, TorchVision, and TorchAudio are Python packages that expand PyTorch's functionality to text, image, and audio data.

A neural network is made up of layers of neurons. Each layer's neurons receive input values and process them through weights and biases. A neuron's output is a weighted sum of inputs plus bias, passed to the next layer's neurons, continuing until the network's final layer is reached.

An activation function transforms a neuron's output, introducing non-linearity into the model's calculations.

Backpropagation trains neural networks by iteratively adjusting neurons' weights and biases to reduce error.

Saturation happens when a neuron's output reaches an extreme beyond which it can’t change, potentially hindering learning. Activation functions like ReLU can help prevent this.

The loss function measures the difference between the model's predicted and actual outputs.

The optimizer adjusts neural network parameters to minimize the loss function during training.

The learning rate determines the optimizer’s step size. A low learning rate can slow down training, while a high one may lead to unstable convergence.

Momentum controls the inertia of an optimizer. Low momentum can cause the optimizer to get stuck in local minima, while high momentum might overshoot the solution.

Transfer learning uses a model trained on one task as a starting point for another task, speeding up training.

Fine-tuning involves freezing early layers and only training later layers, focusing on new output requirements.

Accuracy is a metric that evaluates how well a model fits a dataset, measuring the ratio of correct predictions to total data points.




Importing PyTorch
python
Copy code
# Import the core PyTorch package
import torch

# Import neural network functionality
from torch import nn

# Import functional programming tools
import torch.nn.functional as F

# Import optimization functions
import torch.optim as optim

# Import dataset tools
from torch.utils.data import TensorDataset, DataLoader

# Import evaluation metrics
import torchmetrics
Working with Tensors
python
Copy code
# Create a tensor from a list using tensor()
tnsr = torch.tensor([1, 3, 6, 10])

# Get the data type of tensor elements with .dtype
tnsr.dtype  # Returns torch.int64

# Get the dimensions of the tensor with .size()
tnsr.shape  # Returns torch.Size([4])

# Check the device location of the tensor with .device
tnsr.device  # Returns either CPU or GPU

# Create a tensor of zeros using zeros()
tnsr_zrs = torch.zeros(2, 3)

# Create a random tensor with rand()
tnsr_rndm = torch.rand(size=(3, 4))  # Tensor has 3 rows, 4 columns



Datasets and Dataloaders
python
Copy code
# Create a dataset from a pandas DataFrame using TensorDataset()
X = df[feature_columns].values
y = df[target_column].values
dataset = TensorDataset(torch.tensor(X).float(), torch.tensor(y).float())

# Load the data in batches with DataLoader()
dataloader = DataLoader(dataset, batch_size=n, shuffle=True)
Preprocessing
python
Copy code
# One-hot encode categorical variables with one_hot()
F.one_hot(torch.tensor([0, 1, 2]), num_classes=3)  # Returns a tensor of 0s and 1s
Sequential Model Architecture
python
Copy code
# Create a linear layer with m inputs and n outputs using Linear()
lnr = nn.Linear(m, n)

# Get the weight of the layer with .weight
lnr.weight

# Get the bias of the layer with .bias
lnr.bias

# Create a sigmoid activation layer for binary classification with Sigmoid()
nn.Sigmoid()

# Create a softmax activation layer for multi-class classification with Softmax()
nn.Softmax(dim=-1)

# Create a ReLU activation layer to prevent saturation with ReLU()
nn.ReLU()

# Create a LeakyReLU layer to prevent saturation with LeakyReLU()
nn.LeakyReLU(negative_slope=0.05)

# Create a dropout layer to regularize and reduce overfitting with Dropout()
nn.Dropout(p=0.5)

# Define a sequential model with layers
model = nn.Sequential(
    nn.Linear(n_features, i),
    nn.LeakyReLU(),
    nn.Linear(i, j),  # Input size must match the previous layer's output size
    nn.Linear(j, n_classes),
    nn.Softmax(dim=-1)  # The activation layer should be the last layer
)






Fitting a Model and Calculating Loss
Make Predictions
Fit a model to input data:

python
Copy code
prediction = model(input_data).double()
Define Target Values
Get target values:

python
Copy code
actual = torch.tensor(target_values).double()
Calculate Losses

Mean-Squared Error Loss (for regression):
python
Copy code
mse_loss = nn.MSELoss()(prediction, actual)  # Returns tensor(x)
L1 Loss (for robust regression):
python
Copy code
l1_loss = nn.SmoothL1Loss()(prediction, actual)  # Returns tensor(x)
Binary Cross-Entropy Loss (for binary classification):
python
Copy code
bce_loss = nn.BCELoss()(prediction, actual)  # Returns tensor(x)
Cross-Entropy Loss (for multi-class classification):
python
Copy code
ce_loss = nn.CrossEntropyLoss()(prediction, actual)  # Returns tensor(x)
Backpropagation
Calculate gradients via backpropagation:

python
Copy code
loss.backward()
Working with Optimizers
Define Optimizer
Create a stochastic gradient descent (SGD) optimizer, setting learning rate and momentum:

python
Copy code
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
Update Model Parameters
Update neuron parameters:

python
Copy code
optimizer.step()





The Training Loop
Set Model to Training Mode

python
Copy code
model.train()
Define Loss Criterion and Optimizer

python
Copy code
loss_criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
Loop Through Data Chunks in Training Set

python
Copy code
for data in dataloader:
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
The Evaluation Loop
Set Model to Evaluation Mode

python
Copy code
model.eval()
Define Accuracy Metric

python
Copy code
metric = torchmetrics.Accuracy(task="multiclass", num_classes=3)
Loop Through Data Chunks in Validation Set

python
Copy code
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

# Reset Metric for Next Dataset
metric.reset()
Transfer Learning and Fine-Tuning
Save a Model Layer

python
Copy code
torch.save(layer, 'layer.pth')
Load a Saved Model Layer

python
Copy code
new_layer = torch.load('layer.pth')
Freeze Layer Weights for Fine-Tuning

python
Copy code
for name, param in model.named_parameters():
    if name == "0.weight":
        param.requires_grad = False