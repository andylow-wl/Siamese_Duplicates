import torch
import torch.nn as nn
from torchvision.models import vgg16

class SiameseVGG16(nn.Module):
    def __init__(self):
        super(SiameseVGG16, self).__init__()

        # Setting up the VGG model
        vgg = vgg16(pretrained=False)

        #edit first layer to take 1 channel
        vgg.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        # edit last layer to output 2 classes
        vgg.classifier[6] = nn.Linear(4096, 2)

        # modify dropout layers to lower dropout rate
        for layer in vgg.classifier:
            if isinstance(layer, nn.Dropout):
                layer.p = 0.2

        self.vgg = vgg.features
        self.avgpool = vgg.avgpool
        self.fc = nn.Sequential(*list(vgg.classifier.children())[:-1])


    def forward_once(self, x):
        x = self.vgg(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
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