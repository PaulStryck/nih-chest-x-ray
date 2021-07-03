import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms

def get_resnet_34(num_classes, pretrained = True):
    model = models.resnet34(pretrained=pretrained, progress=True)

    # change the last linear layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def get_resnet_50(num_classes, pretrained: bool = True):
    model = models.resnet50(pretrained=pretrained, progress=True)

    # change the last linear layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def get_effnet_b7(num_classes):
    from efficientnet_pytorch import EfficientNet

    model = EfficientNet.from_pretrained('efficientnet-b7',
                                         num_classes = num_classes)

    return model

def get_effnet_b0(num_classes):
    from efficientnet_pytorch import EfficientNet

    model = EfficientNet.from_pretrained('efficientnet-b0',
                                         num_classes = num_classes)

    return model

def get_googlenet(num_classes, pretrained = True):
    if pretrained:
        model = models.googlenet(pretrained  = pretrained,
                                 progress    = True)
        # change the last linear layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        model = models.googlenet(pretrained  = pretrained,
                                 progress    = True,
                                 num_classes = num_classes)

    return model

def get_densenet161(num_classes, pretrained = True):
    if pretrained:
        model = models.densenet161(pretrained  = pretrained,
                                   progress    = True)
        # change the last linear layer
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        model = models.densenet161(pretrained  = pretrained,
                                   progress    = True,
                                   num_classes = num_classes)

    return model

def get_model(num_classes):
    return get_resnet_50(num_classes)

def get_effnet(num_classes):
    return get_effnet_b7(num_classes)
