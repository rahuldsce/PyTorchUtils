import torch.nn as nn
import torch.nn.functional as F

class Mnist(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First convolutional layer: input size (1, 28, 28), output size (32, 26, 26)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, bias=False)
        # Second convolutional layer: input size (32, 26, 26), output size (64, 24, 24)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=False)
        # First max pooling layer: input size (64, 24, 24), output size (64, 12, 12)
        self.pool1 = nn.MaxPool2d(2,2)
        # Third convolutional layer: input size (64, 12, 12), output size (128, 10, 10)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, bias=False)
        # Fourth convolutional layer: input size (128, 10, 10), output size (256, 8, 8)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, bias=False)
        # Fifth convolutional layer: input size (256, 8, 8), output size (512, 6, 6)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, bias=False)
        # Second max pooling layer: input size (512, 6, 6), output size (512, 3, 3)
        self.pool2 = nn.MaxPool2d(2,2)
        # Global average pooling layer: input size (512, 3, 3), output size (512, 1, 1)
        self.gap = nn.AvgPool2d(6)
        # Fully connected layer 1: input size 512, output size 128
        self.fc1 = nn.Linear(512, 128, bias=False)
        # Fully connected layer 2 (output layer): input size 128, output size 10
        self.fc2 = nn.Linear(128, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Apply first convolutional layer
        x = F.relu(self.pool1(F.relu(self.conv2(x))))  # Apply second convolutional layer and max pooling
        x = F.relu(self.conv3(x))  # Apply third convolutional layer
        x = F.relu(self.conv4(x))  # Apply fourth convolutional layer
        x = F.relu(self.conv5(x))  # Apply fifth convolutional layer
        x = self.gap(x)  # Apply global average pooling
        x = x.view(-1, 512)  # Reshape tensor for fully connected layer
        x = self.fc1(x)  # Apply first fully connected layer
        x = self.fc2(x)  # Apply second fully connected layer (output layer)
        return F.log_softmax(x, dim=1)  # Apply log softmax function to get output probabilities

# Group Normalization
class Cifar10_GN(nn.Module):
  def conv_block (self, in_channels, out_channels, kernel_size, padding = 1) :
        return nn.Sequential(
              nn.Conv2d (in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding, bias = False),
              nn.ReLU(),
              nn.GroupNorm(2, out_channels),
              nn.Dropout(0.1))
    
  def trans_block (self, in_channels, out_channels):
    return nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size = 1, padding = 0, bias = False))
    
  def out_block(self, in_channels, kernel_size = 1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size = kernel_size, padding = 0, bias = False))

  def __init__(self, opts=[]):
        super(Net, self).__init__()
        self.conv1 = self.conv_block(3, 16, 3) #32
        self.conv2 = self.conv_block(16, 32, 3) #32
        self.trans1 = self.trans_block(32, 16) #32
        self.conv3 = self.conv_block(16, 16, 3) #16
        self.conv4 = self.conv_block(16, 32, 3) #16
        self.conv5 = self.conv_block(32, 32, 3) #16
        self.trans2 = self.trans_block(32, 16) #16
        self.conv6 = self.conv_block(16, 16, 3) #8
        self.conv7 = self.conv_block(16, 32, 3) #8
        self.conv8 = self.conv_block(32, 32, 3) #8
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=8))
        self.out = self.out_block(32, 1)
        self.pool = nn.MaxPool2d(2, 2)
        

  def forward(self, x):
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.trans1(x)
      x = self.conv3(x)
      x = self.conv4(x)
      x = self.conv5(x)
      x = self.trans2(x)
      x = self.conv6(x)
      x = self.conv7(x)
      x = self.conv8(x)
      x = self.gap(x)
      x = self.out(x)
      x = x.view(-1, 10)
      return F.log_softmax(x, dim=-1)
