import torch
import torch.nn as nn

class SiameseNetworkAlexNet(nn.Module):
    def __init__(self):
        super(SiameseNetworkAlexNet, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.alexnet_cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Avg adaptive pooling
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # Setting up the Fully Connected Layers
        self.alexnet_fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        )

    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        output = self.alexnet_cnn(x)
        output = self.avgpool(output)
        output = torch.flatten(output, 1)
        output = self.alexnet_fc(output)
        return output
    
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
