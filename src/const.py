#!/usr/bin/env python3

REPO_NAME = 'jinensetpal/sawit'
BUCKET_NAME = 's3://sawit'
DATASOURCE_NAME = 'sawit'
DATASET_NAME = 'sawit'

SPLITS = {'train': 0.8, 'valid': 0.2}
CLASSES = ['Spider', 'Frog', 'Small_mammal', 'Big_mammal', 'Bird', 'Lizard', 'Scorpion']

"""confidence levels guide
high: human/human-verified annotation
medium: high confidence prediction from a strong model
low: low confidence prediction from a weak model
"""
