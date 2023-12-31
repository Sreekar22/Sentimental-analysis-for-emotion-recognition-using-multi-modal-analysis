# -*- coding: utf-8 -*-
"""Multimodal_main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1semE0Fo32sF4wZ1rMeVTr9bmQRDYIQds
"""

!pip install -q condacolab
import condacolab
condacolab.install()

!conda --version

!which conda

from google.colab import drive
drive.mount('/content/drive')

import os, sys
import glob
import pickle
import numpy as np
import pandas as pd
import cv2
from scipy.io import wavfile
from tqdm import tqdm



def read_video(file_name):
    vidcap = cv2.VideoCapture(file_name)

    # Read FPS
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
    else :
        fps = vidcap.get(cv2.CAP_PROP_FPS)

    # Read image data
    success, image = vidcap.read()
    images = []
    while success:
        images.append(image)
        success, image = vidcap.read()
    return np.stack(images), fps

def parse_evaluation_transcript(eval_lines, transcript_lines):
    metadata = {}

    # Parse Evaluation
    for line in eval_lines:
        if line.startswith('['):
            tokens = line.strip().split('\t')
            time_tokens = tokens[0][1:-1].split(' ')
            start_time, end_time = float(time_tokens[0]), float(time_tokens[2])
            uttr_id, label = tokens[1], tokens[2]
            metadata[uttr_id] = {'start_time': start_time, 'end_time': end_time, 'label': label}

    # Parse Transcript
    trans = []
    for line in transcript_lines:
        tokens = line.split(':')
        uttr_id = tokens[0].split(' ')[0]
        if '_' not in uttr_id:
            continue
        text = tokens[-1].strip()
        try:
            metadata[uttr_id]['text'] = text
        except KeyError:
            print(f'KeyError: {uttr_id}')
    return metadata

def retrieve_audio(signal, sr, start_time, end_time):
    start_idx = int(sr * start_time)
    end_idx = int(sr * end_time)
    audio_segment = signal[start_idx:end_idx]
    return audio_segment, sr

def retrieve_video(frames, fps, start_time, end_time):
    start_idx = int(fps * start_time)
    end_idx = int(fps * end_time)
    images = frames[start_idx:end_idx,:,:,:]
    return images, fps

def dump_image_audio(uttr_id, audio_segment, sr, img_segment, img_segment_L, img_segment_R, fps, out_path='./', grayscale=False):
    out_path = f'{out_path}/{"_".join(uttr_id.split("_")[:2])}'
    if not os.path.exists(f'./{out_path}/{uttr_id}'):
        os.makedirs(f'./{out_path}/{uttr_id}')
    wavfile.write(f'./{out_path}/{uttr_id}/audio.wav', sr, audio_segment)
    wavfile.write(f'./{out_path}/{uttr_id}/audio_L.wav', sr, audio_segment[:,0])
    wavfile.write(f'./{out_path}/{uttr_id}/audio_R.wav', sr, audio_segment[:,1])
    for i in range(img_segment.shape[0]):
#         cv2.imwrite(f'./{out_path}/{uttr_id}/image_{i}.jpg', img_segment[i,:,:,:])
        imgL = img_segment_L[i,:,:,:]
        imgR = img_segment_R[i,:,:,:]
        if grayscale:
            imgL = rgb2gray(imgL)
            imgR = rgb2gray(imgR)
        cv2.imwrite(f'./{out_path}/{uttr_id}/image_L_{i}.jpg', imgL)
        cv2.imwrite(f'./{out_path}/{uttr_id}/image_R_{i}.jpg', imgR)

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def crop(imgs, target_size=224):
    # imgs.shape = (180, 480, 360, 3)
    _, h, w, _ = imgs.shape
    offset_h = (h - target_size) // 2
    offset_w = (w - target_size) // 2
    imgs = imgs[:, offset_h:-offset_h, offset_w:-offset_w, :]
    return imgs

# Process multimodal data over all sessions
# NOTE: This might take several hours to run, the time listed on this cell is for processing 5 label files
output_path =  '/content/drive/MyDrive/sample_folder_preprocess/IEMOCAP_PREPROCESS'


if not os.path.exists(output_path):
    os.makedirs(output_path)

all_metas = {}
for base_path in glob.glob('/content/drive/MyDrive/sample folder preprocess*'):
    avi_path = '/content/drive/MyDrive/sample folder preprocess/avi'
    script_path = '/content/drive/MyDrive/sample folder preprocess/transcriptions'
    wav_path = '/content/drive/MyDrive/sample folder preprocess/wav'
    label_path = '/content/drive/MyDrive/sample folder preprocess/Emo_evaluation'

    for eval_fname in tqdm(glob.glob(f'{label_path}/*.txt')):
        avi_fname = f'{avi_path}/{eval_fname.split("/")[-1].replace(".txt", ".avi")}'
        wav_fname = f'{wav_path}/{eval_fname.split("/")[-1].replace(".txt", ".wav")}'
        script_fname = f'{script_path}/{eval_fname.split("/")[-1]}'

        eval_lines = open(eval_fname).readlines()
        transcript_lines = open(script_fname).readlines()
        sr, signal  = wavfile.read(wav_fname)

        images, fps = read_video(avi_fname)

        # Retrieve uttr_id, label, time, and transcript
        metas = parse_evaluation_transcript(eval_lines, transcript_lines)

        for uttr_id, metadata in metas.items():
            # Retrieve and Store Audio
            audio_segment, sr = retrieve_audio(signal, sr, metadata['start_time'], metadata['end_time'])
            metadata['sr'] = sr

            img_segment, fps = retrieve_video(images, fps, metadata['start_time'], metadata['end_time'])
            img_segment_L, img_segment_R = img_segment[:,:,:img_segment.shape[2] // 2,:], img_segment[:,:,img_segment.shape[2] // 2:,:]
            img_segment_L = crop(img_segment_L)
            img_segment_R = crop(img_segment_R)
            metadata['fps'] = fps

            dump_image_audio(uttr_id, audio_segment, sr, img_segment, img_segment_L, img_segment_R, fps, out_path=output_path)

        # Update all metas
        all_metas.update(metas)
pickle.dump(all_metas, open(f'{output_path}/meta.pkl','wb'))

!pip install torch torchaudio torchvision transformers  facenet-pytorch

!git clone https://github.com/facebookresearch/SparseConvNet

# Commented out IPython magic to ensure Python compatibility.
# %cd SparseConvNet/

!bash develop.sh

!git clone https://github.com/wenliangdai/Multimodal-End2end-Sparse.git

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/SparseCon/

!pip install tabulate
!pip install scikit-learn

!python /content/SparseConvNet/Multimodal-End2end-Sparse/main.py

!python main.py -lr=5e-5 -ep=40 -mod=tav -bs=8 --img-interval=500 --datapath='/content/drive/MyDrive/sample folder preprocess' --dataset='iemocap' --optim='adam' --early-stop=6 --loss=bce --cuda=3 --model=mme2e --num-emotions=6 --trans-dim=64 --trans-nlayers=4 --trans-nheads=4 --text-lr-factor=10 --text-model-size=base --text-max-len=100
--fusion='early' --hfc-sizes=[300, 144, 35] --audio-feautre-type=0
