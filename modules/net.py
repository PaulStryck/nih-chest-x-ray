import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms


def get_model(num_classes):
    model = models.resnet50(pretrained=True, progress=True)

    # change the last linear layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def get_effnet(num_classes):
    from efficientnet_pytorch import EfficientNet

    model = EfficientNet.from_pretrained('efficientnet-b7',
                                         num_classes = num_classes)

    return model

def get_effnet_b0(num_classes):
    from efficientnet_pytorch import EfficientNet

    model = EfficientNet.from_pretrained('efficientnet-b0',
                                         num_classes = num_classes)

    return model
