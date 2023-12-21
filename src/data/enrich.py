#!/usr/bin/env python3

from dagshub.data_engine import datasources
from src import const

def enrich(row): 
    file = row['path'].split('/')[-1][:-5]

    row['type'] = 'image'
    row['labels'] = "labels/VOC_format/{file}.txt"
    if not file.startswith('IMG'):
        row['video'], row['date'], row['time'] = file.split('--')
        row['video'] = row['video'][-3:]
        row['time'] = row['time'][:8]
    
    return row

def preprocess():
    if not len(datasources.get_datasources(const.REPO_NAME)):
        ds = datasources.create_from_bucket(repo=const.REPO_NAME,
                                            name=const.DATASOURCE_NAME,
                                            path=const.BUCKET_NAME)
    else: ds = datasources.get_datasource(const.REPO_NAME, name=const.DATASOURCE_NAME)

    df = ds['path'].contains('images').all().dataframe.apply(enrich, axis=1)
    ds.upload_metadata_from_dataframe(df)

    (ds['type'] == 'image').save_dataset(const.DATASET_NAME)

if __name__ == '__main__':
    preprocess()
