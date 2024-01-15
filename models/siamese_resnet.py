import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class SiameseResNet(nn.Module):
    def __init__(self):
        super(SiameseResNet, self).__init__()

        # Create a ResNet18 model (pre-trained or from scratch)
        resnet = resnet18(pretrained=False)

        # Modify the first convolutional layer to handle single-channel (grayscale) images
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Use the rest of the ResNet layers (excluding the last FC layer)
        self.resnet = nn.Sequential(*list(resnet.children())[1:-1])

        # Modify the final FC layer to output the desired embedding size
        in_features = resnet.fc.in_features

        # Additional linear layers
        self.fc1 = nn.Linear(in_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward_once(self, x):
        # Forward pass through the modified ResNet
        x = F.relu(self.conv1(x))
        x = self.resnet(x)
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def forward(self, input1=None, input2=None, input3=None):
        if input3==None:
            output1 = self.forward_once(input1)
            output2 = self.forward_once(input2)
            return output1,output2
        else:
            output1 = self.forward_once(input1)
            output2 = self.forward_once(input2)
            output3 = self.forward_once(input3)
            return output1,output2,output3