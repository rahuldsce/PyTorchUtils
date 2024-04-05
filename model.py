import torch.nn as nn
import torch.nn.functional as F

class Mnist(nn.Module):
    def __init__(self):
        super(Mnist, self).__init__()
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

class Cifar10_BN(nn.Module):
  def conv_block (self, in_channels, out_channels, kernel_size, padding = 1) :
        return nn.Sequential(
              nn.Conv2d (in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding, bias = False),
              nn.ReLU(),
              nn.BatchNorm2d(out_channels),
              nn.Dropout(0.1))

  def trans_block (self, in_channels, out_channels):
    return nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size = 1, padding = 0, bias = False))

  def out_block(self, in_channels, kernel_size = 1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size = kernel_size, padding = 0, bias = False))

  def __init__(self, opts=[]):
        super(Cifar10_BN, self).__init__()
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
        super(Cifar10_GN, self).__init__()
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

# Layer Normalization 
class Cifar10_LN(nn.Module):
  def conv_block (self, in_channels, out_channels, kernel_size, input_shape=[32, 32], padding = 1) :
        return nn.Sequential(
              nn.Conv2d (in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding, bias = False),
              nn.ReLU(),
              nn.LayerNorm([out_channels] + input_shape),
              nn.Dropout(0.1))

  def trans_block (self, in_channels, out_channels):
    return nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size = 1, padding = 0, bias = False))

  def out_block(self, in_channels, kernel_size = 1):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size = kernel_size, padding = 0, bias = False))

  def __init__(self, opts=[]):
        super(Cifar10_LN, self).__init__()
        self.conv1 = self.conv_block(3, 16, 3, input_shape=[32,32]) #32
        self.conv2 = self.conv_block(16, 32, 3, input_shape=[32,32]) #32
        self.trans1 = self.trans_block(32, 16) #32
        self.conv3 = self.conv_block(16, 16, 3, input_shape=[16,16]) #16
        self.conv4 = self.conv_block(16, 32, 3, input_shape=[16,16]) #16
        self.conv5 = self.conv_block(32, 32, 3, input_shape=[16,16]) #16
        self.trans2 = self.trans_block(32, 16) #16
        self.conv6 = self.conv_block(16, 16, 3, input_shape=[8,8]) #8
        self.conv7 = self.conv_block(16, 32, 3, input_shape=[8,8]) #8
        self.conv8 = self.conv_block(32, 32, 3, input_shape=[8,8]) #8
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

import torch.nn.functional as F
import torch.nn as nn

class Cifar10_S9(nn.Module):
    def conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation = 1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation = dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def sep_conv_block(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=1, groups=in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def out_block(self, in_channels, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=1)
        )

    def __init__(self, num_classes=10):
        super(Cifar10_S9, self).__init__()
        self.conv1 = self.conv_block(3, 64)
        self.conv2 = self.conv_block(64, 64, stride=2)
        self.conv3 = self.conv_block(64, 64, stride=2)
        self.sep_conv = self.sep_conv_block(64, 64)
        self.dilated_conv = self.conv_block(64, 64, kernel_size=3, dilation=2)
        self.conv4 = self.conv_block(64, 32)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out = self.out_block(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.sep_conv(x)
        x = self.dilated_conv(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = self.out(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

class CIFAR10_S10_RES_NET(nn.Module):
    
    def prep_block (self, in_channels, out_channels, kernel_size, padding = 1, stride=1) :
        return nn.Sequential(
              nn.Conv2d (in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding, stride=stride, bias = False),
              nn.BatchNorm2d(out_channels),
              nn.ReLU())
        
    def conv_block (self, in_channels, out_channels, kernel_size, padding = 1, stride=1) :
        return nn.Sequential(
              nn.Conv2d (in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding, stride=stride, bias = False),
              nn.MaxPool2d(2, 2),
              nn.BatchNorm2d(out_channels),
              nn.ReLU())
        
    def res_block (self, in_channels, out_channels, kernel_size, padding = 1, stride=1) :
        return nn.Sequential(
              nn.Conv2d (in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding, stride=stride, bias = False),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(),
              nn.Conv2d (in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, padding = padding, stride=stride, bias = False),
              nn.BatchNorm2d(out_channels),
              nn.ReLU())

    def __init__(self, opts=[]):
        super(CIFAR10_DAVID_RES_NET, self).__init__()
        
        # Prep Layer
        self.prep_layer = self.prep_block(3, 64, 3)
        
        # Layer 1
        self.conv_l1 = self.conv_block(64, 128, 3)
        self.res_block_l1 = self.res_block(128, 128, 3)
        
        # Layer 2
        self.conv_l2 = self.conv_block(128, 256, 3)
        
        # Layer 3
        self.conv_l3 = self.conv_block(256, 512, 3)
        self.res_block_l3 = self.res_block(512, 512, 3)
        
        # Pool
        self.pool = nn.MaxPool2d(4, 4)
        
        # FC Layer
        self.FC = nn.Linear(512, 10)

    def forward(self, x):
        # Prep Layer
        x = self.prep_layer(x)
        
        # Layer 1
        x = self.conv_l1(x)
        R1 = self.res_block_l1(x)
        x = x + R1

        # Layer 2
        x = self.conv_l2(x)
        
        # Layer 3
        x = self.conv_l3(x)
        R2 = self.res_block_l3(x)
        x = x + R2
        
        # Pool
        x = self.pool(x)
        
        # FC Layer
        x = self.FC(x.view(x.size(0), -1))

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
