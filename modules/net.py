import torch
import copy
from torch import Tensor
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


class LargeResNet(nn.Module):
    def __init__(self, num_classes, pretrained = True):
        super(LargeResNet, self).__init__()
        _model = get_resnet_50(num_classes, pretrained = pretrained)

        self.conv1    = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                  bias=False)
        self.bn1      = copy.deepcopy(_model.bn1)
        self.relu     = copy.deepcopy(_model.relu)
        self.maxpool  = copy.deepcopy(_model.maxpool)

        self.layer1   = copy.deepcopy(_model.layer1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer2   = copy.deepcopy(_model.layer2)
        self.layer3   = copy.deepcopy(_model.layer3)
        self.layer4   = copy.deepcopy(_model.layer4)

        self.avgpool  = copy.deepcopy(_model.avgpool)
        # torch.flatten(x, 1)
        self.fc       = copy.deepcopy(_model.fc)


        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
