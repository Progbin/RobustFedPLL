import torch.nn as nn

class CNN_2Conv(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_2Conv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.projector = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.projector(x)
        x = self.classifier(x)
        return x
