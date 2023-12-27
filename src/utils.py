#!/usr/bin/env python3

from src import const
import torch
import uuid


def warmup(model):
    x = torch.empty(1, *const.IMAGE_SHAPE, dtype=torch.half, device=const.DEVICE)

    try:
        with torch.no_grad():
            model(x)
    except RuntimeError: warmup(model)

    return model


# adapted from https://dagshub.com/Dean/COCO_1K/src/main/utils/yolo_converter.py
def create_bbox(line, voc=False):
    if voc:
        label_id, x, y, width, height = line.split()
        label_id = const.CLASSES.index(label_id)
        confidence = 1.0
    else: x, y, width, height, confidence, label_id = line.split()
    x, y, width, height, confidence = (float(x),
                                       float(y),
                                       float(width),
                                       float(height),
                                       float(confidence))

    return {"id": uuid.uuid4().hex[0:10],
            "type": "rectanglelabels",
            "value": {"x": (x - width / 2) * 100,
                      "y": (y - height / 2) * 100,
                      "width": width * 100,
                      "height": height * 100,
                      "rotation": 0,
                      "rectanglelabels": [const.CLASSES[int(label_id)]]},
            "confidence": confidence,
            "to_name": 'image',
            "from_name": 'label',
            "image_rotation": 0,
            "original_width": const.IMAGE_SIZE[0],
            "original_height": const.IMAGE_SIZE[1]}
