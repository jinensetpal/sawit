#!/usr/bin/env python3

from src import const
import torchvision
import torch


def get_tensorizers():
    return image_norm, target


def image_norm(filepath):
    return torchvision.io.read_image(filepath) / 255


def target(filepath):
    targets = []
    with open(filepath, 'r') as file:
        for line in file.readlines():
            target = line.split()
            targets.append([torch.tensor(const.CLASSES.index(target[0])),
                            torch.tensor([int(x) for x in target[1:5]])])

    return targets
