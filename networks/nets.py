import torch
import torch.nn as nn
import torchvision.models.segmentation as models
from collections import OrderedDict


class NetEnd(nn.Module):
    def __init__(self, num_classes:int):
        super(NetEnd, self).__init__()
        self.num_classes = num_classes
        self.fc_net1 = nn.Conv2d(21, self.num_classes, kernel_size=1, stride=1)
        assert self.num_classes > 0, 'The number of classes must be a positive integer.'
        if self.num_classes > 1:
            self.final = nn.Softmax()
        else:
            self.final = nn.Sigmoid()

    def forward(self, x):
        out = self.fc_net1(x)
        out = self.final(out)
        return out

class ClassifierEnd(nn.Module):
    def __init__(self, num_classes:int):
        super(ClassifierEnd, self).__init__()
        self.num_classes = num_classes
        self.fc_net1 = nn.Conv2d(21, self.num_classes, kernel_size=1, stride=1)
        self.fc_net2 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=1, stride=1)
        self.fc_net3 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=1, stride=1)
        self.fc_net4 = nn.Conv2d(self.num_classes, self.num_classes, kernel_size=1, stride=1)
        assert self.num_classes > 0, 'The number of classes must be a positive integer.'
        if self.num_classes > 1:
            self.final = nn.Softmax()
        else:
            self.final = nn.Sigmoid()

    def forward(self, x):
        out = self.fc_net1(x)
        out = self.fc_net2(out)
        out = self.fc_net3(out)
        out = self.fc_net4(out)
        out = self.final(out)

        return out



class ResNet101_DeeplabV3(nn.Module):
    def __init__(self, end_module, keyword='out', pretrain=False):
        super(ResNet101_DeeplabV3, self).__init__()
        self.deeplab = models.deeplabv3_resnet101(pretrained=pretrain,
                                                  progress=True,
                                                  num_classes=21)
        self.end_module = end_module
        self.output = None
        self.key = keyword

    def forward(self, x):
        if isinstance(x, OrderedDict):
            self.output = self.deeplab(x[self.key])
        else:
            self.output = self.deeplab.forward(x)

        if isinstance(self.output, OrderedDict):
            self.output = self.end_module.forward(self.output[self.key])
        else:
            self.output = self.end_module.forward(self.output)

        return self.output


class ResNet50_DeeplabV3(nn.Module):
    def __init__(self, end_module, keyword='out', pretrain=False):
        super(ResNet50_DeeplabV3, self).__init__()
        self.deeplab = models.deeplabv3_resnet50(pretrained=pretrain,
                                                 progress=True,
                                                 num_classes=21)
        self.end_module = end_module
        self.output = None
        self.key = keyword

    def forward(self, x):
        if isinstance(x, OrderedDict):
            self.output = self.deeplab(x[self.key])
        else:
            self.output = self.deeplab.forward(x)

        if isinstance(self.output, OrderedDict):
            self.output = self.end_module.forward(self.output[self.key])
        else:
            self.output = self.end_module.forward(self.output)

        return self.output


class ResNet101_FCN(nn.Module):
    def __init__(self, end_module, keyword='out', pretrain=False):
        super(ResNet101_FCN, self).__init__()
        self.resnet = models.fcn_resnet101(pretrained=pretrain,
                                          progress=True,
                                          num_classes=21)
        self.end_module = end_module
        self.output = None
        self.key = keyword

    def forward(self, x):
        if isinstance(x, OrderedDict):
            self.output = self.resnet(x[self.key])
        else:
            self.output = self.resnet.forward(x)

        if isinstance(self.output, OrderedDict):
            self.output = self.end_module.forward(self.output[self.key])
        else:
            self.output = self.end_module.forward(self.output)

        return self.output














