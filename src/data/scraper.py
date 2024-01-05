#!/usr/bin/env python3

from src.utils import warmup, create_bbox
from src import const

from utils.general import non_max_suppression
from dagshub.data_engine import datasources
from dagshub import get_repo_bucket_client
import imageio.v3 as iio
from pathlib import Path
import multiprocessing
from PIL import Image
from glob import glob
import pandas as pd
import torchvision
import youtube_dl
import datetime
import shutil
import random
import torch
import pafy
import cv2
import sys
import os


# TODO: multiproc
def stream(target):
    cap = cv2.VideoCapture(pafy.new(target).getbest(preftype='mp4').url)
    while True:
        _, img = cap.read()
        yield img


def single_download(target):
    with youtube_dl.YoutubeDL({'outtmpl': (const.STAGING_DIR / f"{target.split('=')[-1]}--{datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')}").as_posix()}) as ydl: ydl.download([target])


def download(targets):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(single_download, targets)


def annotate():
    ds = datasources.get_datasource(repo=const.REPO_NAME, name=const.DATASET_NAME)
    bucket = get_repo_bucket_client(const.REPO_NAME, flavor='boto')

    model = torch.load(const.MODEL_DIR / const.ANNOTATOR_MODEL / 'best.pt')['model'].to(const.DEVICE)
    model.eval()
    model = warmup(model)

    metadata = []
    for target in glob(Path(const.STAGING_DIR / '*.mkv').as_posix()):
        video, date, time = target[:-4].split('/')[-1].split('--')
        n_frames = len(ds['video'].contains(video).all())
        for idx, frame in enumerate(iio.imiter(target, plugin='pyav')):
            if idx % 20: continue
            framename = f'{video}--{date}--{time}--{int(n_frames+idx/20+1)}.png'

            img = Image.fromarray(frame).resize(const.IMAGE_SIZE[::-1])
            img.save('frame.png')
            bucket.upload_file('frame.png', const.REPO_NAME.split('/')[0], 'images' / 'augmented' / framename)

            labels = []
            with torch.no_grad():
                for detection in non_max_suppression(model(torchvision.transforms.functional.pil_to_tensor(img).to(torch.half).unsqueeze(0).to(const.DEVICE)),
                                                     iou_thres=0, conf_thres=0.99, max_det=10)[0].tolist():
                    labels.append(create_bbox(detection))

            metadata.append({'path': (Path('images') / 'augmented' / framename).as_posix(),
                             'date': date,
                             'time': time,
                             'video': video,
                             'type': 'image',
                             'annotator': const.ANNOTATOR_MODEL,
                             'split': random.choices(list(const.SPLITS.keys()), weights=list(const.SPLITS.values()))[0],
                             'labels': str(labels)})

    shutil.rmtree(const.STAGING_DIR)
    ds.upload_metadata_from_dataframe(pd.DataFrame(metadata), path_column='path')


if __name__ == '__main__':
    dagshub.auth.add_app_token(os.getenv('TOKEN', ''))

    if sys.argv[1] == 'download':
        download(const.TARGETS)
    elif sys.argv[1] == 'annotate':
        if len(sys.argv) == 2: const.STAGING_DIR = Path(os.environ['CODEBUILD_SRC_DIR_staging'])
        annotate()
