import torch.nn as nn


def cross_entropy_2d(output, target):

    target = target.squeeze(1)

    criterion = nn.CrossEntropyLoss()

    loss = criterion(output, target.long())

    return loss


def binary_entropy_2d(output, target):
    # for binary segmentation
    output = output.squeeze(1)
    target = target.squeeze(1)

    criterion = nn.BCELoss()

    loss = criterion(output, target)

    return loss