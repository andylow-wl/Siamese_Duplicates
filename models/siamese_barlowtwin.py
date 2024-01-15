import torch
import torch.nn as nn

class SiameseNetworkBarlowTwins(nn.Module):

    def __init__(self):
        super(SiameseNetworkBarlowTwins, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 2)
        )

    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similarity
        output = self.cnn1(x)
        unscaled_features = output.view(output.size()[0], -1)  # Unscaled feature vectors
        output = self.fc1(unscaled_features)
        return output, unscaled_features  # Return both the final output and unscaled feature vectors

    def forward(self, input1, input2):
        # In this function, we pass in both images and obtain both vectors
        # which are returned
        output1, z1 = self.forward_once(input1)
        output2, z2 = self.forward_once(input2)

        return output1, output2, z1, z2  # Return both the final outputs and unscaled feature vectors