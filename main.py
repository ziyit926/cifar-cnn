import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import train
import test

# Ensure the training will only happen in CPU.
device = torch.device('cpu')

# Data preprocessing
all_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4), # Random crop with padding
                                     transforms.RandomHorizontalFlip(), # Random horizontal flip
                                     transforms.ToTensor(), # Convert image to tensors
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalise the image data
                                     ])

# Download and create training dataset and testing dataset with above transforms applied
train_dataset = torchvision.datasets.CIFAR10(root = './data',
                                             train = True,
                                             transform = all_transforms,
                                             download = True)

test_dataset = torchvision.datasets.CIFAR10(root = './data',
                                            train = False,
                                            transform = all_transforms,
                                            download=True)

# Number of samples pre batch throw to the model during training and testing
batch_size = 64

# DataLoader for training data and testing data loads data in batches, shuffles for randomness
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = False)

# Printing a batch to find out the input and output tensor sizes for initialising the neural network
for inputs, labels in test_loader:
    print(inputs.shape) # 64 (batch size), 3 (RGB color), 32, 32
    print(labels.shape) # 64 (batch size)
    break

# Define Convolutional Neural Network model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Create the first conv layer by using the input and output from above
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU() # Activation function
        self.pool1 = nn.MaxPool2d(2, 2) # Max pooling halves the spatial size from 32x32 to 16x16

        # Create second conv with 32 input channels, 64  output filters
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2) # Max pooling halves the spatial size from 16x16 to 8x8

        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu3 = nn.ReLU()

        # Create final fully connected layer outputs 10 values, one for each class
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Applying first conv, activation, and pooling
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Applying second conv, activation, and pooling
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten the feature maps into vector to throw into fully connected layer
        x = torch.flatten(x, start_dim=1)

        # Applying fully connected layers and activation
        x = self.fc1(x)
        x = self.relu3(x)

        # Applying output layer
        x = self.fc2(x)
        return x

# Instantiate the model and move it to the device
model = ConvNet().to(device)
print(model) # Print model architecture for checking

# Count total number of parameters (weights + biases)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}\n")

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the number of full passes through the training data
epochs = 20

train.model = model
train.loss_fn = loss_fn
train.optimizer = optimizer
train.device = device
train.train_loader = train_loader

test.model = model
test.loss_fn = loss_fn
test.device = device
test.test_loader = test_loader

for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    train.train()
    test.test()

# Save the trained model weights to a file
PATH = "./cifar_net.pth"
torch.save(model.state_dict(), PATH)

# Load the model weights
model_saved = ConvNet().to(device)
model_saved.load_state_dict(torch.load(PATH, weights_only=True))
model_saved.eval()

print("Saved model on test set:")
test.test()