import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print('Training samples:',len(train_data))
print('Test samples:',len(test_data))

image, label = train_data[0]
print("Image shape: ",image.shape)
print("Label: ",label)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

image, label = next(iter(train_loader))
print('Batch Shape: ',image.shape)
print('Label Shape: ',label.shape)
