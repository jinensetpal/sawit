#!/usr/bin/env python3

from pathlib import Path
import torch
import sys

# directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
YOLO_DIR = BASE_DIR / 'src' / 'model' / 'yolo'
STAGING_DIR = DATA_DIR / 'staging'
STAGING_DIR.mkdir(parents=True, exist_ok=True)
sys.path.append(YOLO_DIR.as_posix())

# remotes
REPO_NAME = 'jinensetpal/sawit'
MLFLOW_TRACKING_URI = f'https://dagshub.com/{REPO_NAME}.mlflow'
BUCKET_NAME = 's3://sawit'
DATASOURCE_NAME = 'sawit'
DATASET_NAME = 'sawit'

# dataset
N_CHANNELS = 3
IMAGE_SIZE = (640, 480)
IMAGE_SHAPE = (N_CHANNELS, *IMAGE_SIZE)
LABEL_KEYS = ['class', 'x', 'y', 'w', 'h', 'confidence']
CLASSES = ['Frog', 'Lizard', 'Bird', 'Small_mammal', 'Big_mammal', 'Spider', 'Scorpion']

# training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SPLITS = {'train': 0.8, 'valid': 0.2}

# automated enrichment
# TARGETS = ['https://www.youtube.com/watch?v=dIChLG4_WNs',
#            'https://www.youtube.com/watch?v=ydYDqZQpim8',
#            'https://www.youtube.com/watch?v=39uYW98qOV0',
#            'https://www.youtube.com/watch?v=F0GOOP82094',
#            'https://www.youtube.com/watch?v=cKe0WSZKYgQ']
TARGETS = ['https://www.youtube.com/watch?v=9jZH_5ZBuQQ',]
ANNOTATOR_MODEL = 'yolo5s'
