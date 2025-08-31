import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, feature_dim=100):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=True)

        num_ftrs = self.resnet.fc.in_features
        print("num_ftrs", num_ftrs)
        self.resnet.fc = nn.Identity()

        self.classifier = nn.Linear(num_ftrs, feature_dim)


    def forward(self, x):
        features = self.resnet(x)
        output = self.classifier(features)
        return output


