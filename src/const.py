#!/usr/bin/env python3

"""confidence levels guide
high: human/human-verified annotation
medium: high confidence prediction from a strong model
low: low confidence prediction from a weak model
"""

from pathlib import Path
import torch
import sys

# directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
YOLO_DIR = BASE_DIR / 'src' / 'model' / 'yolo'
sys.path.append(YOLO_DIR.as_posix())

# remotes
REPO_NAME = 'jinensetpal/saw-et-al'
MLFLOW_TRACKING_URI = f'https://dagshub.com/{REPO_NAME}.mlflow'
BUCKET_NAME = 's3://saw-et-al'
DATASOURCE_NAME = 'saw-et-al'
DATASET_NAME = 'saw-et-al'

# dataset
N_CHANNELS = 3
IMAGE_SIZE = (640, 480)
IMAGE_SHAPE = (N_CHANNELS, *IMAGE_SIZE)
CLASSES = ['Frog', 'Lizard', 'Bird', 'Small_mammal', 'Big_mammal', 'Spider', 'Scorpion']

# training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SPLITS = {'train': 0.8, 'valid': 0.2}

# automated enrichment
TARGETS = ['https://www.youtube.com/live/dIChLG4_WNs',
           'https://www.youtube.com/live/ydYDqZQpim8',
           'https://www.youtube.com/live/39uYW98qOV0',
           'https://www.youtube.com/live/F0GOOP82094',
           'https://www.youtube.com/live/cKe0WSZKYgQ']
ANNOTATOR_MODEL = 'yolo5s'
