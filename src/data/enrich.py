#!/usr/bin/env python3

from dagshub.data_engine import datasources
from src.utils import create_bbox
from src import const
import random


def enrich(row):
    splits = row['path'].split('/')
    file = splits[-1][:-5]

    row['split'] = splits[1]
    if row['split'] == 'train': row['split'] = random.choices(list(const.SPLITS.keys()), weights=list(const.SPLITS.values()))

    row['type'] = 'image'
    row['labels'] = f'labels/VOC_format/{file}.txt'
    row['annotator'] = 'human'
    if not file.startswith('IMG'):
        row['video'], row['date'], row['time'] = file.split('--')
        row['video'] = row['video'][-3:]
        row['time'] = row['time'][:8]

    return row


def encoded_json(filepath):
    return str([create_bbox(line, voc=True) for line in open(filepath, 'r').readlines()])


def preprocess():
    if not len(datasources.get_datasources(const.REPO_NAME)):
        ds = datasources.create_from_bucket(repo=const.REPO_NAME,
                                            name=const.DATASOURCE_NAME,
                                            path=const.BUCKET_NAME)
    else: ds = datasources.get_datasource(const.REPO_NAME, name=const.DATASOURCE_NAME)

    df = ds['path'].contains('images').all().dataframe.apply(enrich, axis=1)
    ds.upload_metadata_from_dataframe(df, path_column='labels')

    q = (ds['type'] == 'image')
    df['labels'] = list(q.all().as_ml_dataset('torch', tensorizers=[encoded_json,]))
    df['labels'] = df['labels'].apply(lambda x: x[0])

    ds.upload_metadata_from_dataframe(df, path_column='path')
    q.save_dataset(const.DATASET_NAME)


if __name__ == '__main__':
    preprocess()
