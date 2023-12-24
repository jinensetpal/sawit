#!/usr/bin/env python3

from utils.general import non_max_suppression
from dagshub import get_repo_bucket_client
from dagshub.data_engine import datasets
from src.utils import warmup
import imageio.v3 as iio
from pathlib import Path
from PIL import Image
from src import const
from glob import glob
import pandas as pd
import youtube_dl
import datetime
import shutil
import random
import torch


def download(targets):
    for target in targets:
        with youtube_dl.YoutubeDL({'outtmpl': (const.DATA_DIR / 'staging' / f"{target.split('/')[-1]}--{datetime.datetime.now().strftime('%Y-%M-%d--%H-%M-%S')}").as_posix()}) as ydl: ydl.download([target])


def scrape():
    ds = datasets.get_dataset(repo=const.REPO_NAME, name=const.DATASET_NAME)
    bucket = get_repo_bucket_client(const.REPO_NAME, flavor='boto')

    model = torch.load(const.MODEL_DIR / const.ANNOTATOR_MODEL / 'best.pt')['model'].to(const.DEVICE)
    model.eval()
    model = warmup(model)

    metadata = []
    for target in glob(Path(const.DATA_DIR / 'staging').as_posix()):
        n_frames = ds['video'].contains(target.split('/')[-1].split('--')[0])
        video, date, time = target.split('/')[-1].split('--')
        for idx, frame in iio.imiter(target, plugin='pyav'):
            if idx % 20: continue
            framename = f'{video}-{date}-{time}-{n_frames+idx+1}.png'

            Image.fromarray(frame).save('frame.png')
            bucket.upload_file('frame.png', const.REPO_NAME, const.DATA_DIR / 'images' / 'augmented' / f'{framename}.png')

            with open('labels.txt', 'w'), torch.no_grad() as labels:
                for detection in non_max_suppression(model(torch.tensor([frame], dtype=torch.half)), iou_thres=0, conf_thres=0.99, max_det=10)[0].tolist():
                    labels.write(' '.join([const.CLASSES[int(detection[-1])], *detection[:-1]]) + '\n')
                bucket.upload_file('labels.txt', const.REPO_NAME, const.DATA_DIR / 'labels' / f'{framename}.txt')

            metadata.append({'path': (const.DATA_DIR / 'images' / 'augmented' / f'{framename}.png').as_posix(),
                             'date': date,
                             'time': time,
                             'video': video,
                             'type': 'image',
                             'annotator': const.ANNOTATOR_MODEL,
                             'split': random.choices(list(const.SPLITS.keys()), weights=list(const.SPLITS.values())),
                             'labels': (const.DATA_DIR / 'labels' / f'{framename}.txt').as_posix()})

    shutil.rmtree(const.DATA_DIR / 'staging')
    ds.upload_metadata_from_dataframe(pd.DataFrame(metadata), path_column='path')


if __name__ == '__main__':
    download(const.TARGETS)
    scrape()
