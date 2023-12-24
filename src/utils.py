#!/usr/bin/env python3

from src import const
import torch


def warmup(model):
    x = torch.empty(1, *const.IMAGE_SHAPE, dtype=torch.half, device=const.DEVICE)

    try:
        with torch.no_grad():
            model(x)
    except RuntimeError: warmup(model)

    return model
