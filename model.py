import torch.nn as nn
from torchvision import models

class ChestXrayModel(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()

        self.model = models.densenet121(weights="IMAGENET1K_V1")
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)
