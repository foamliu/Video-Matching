import json
import os
import pickle

import cv2 as cv
import numpy as np
import torch
from tqdm import tqdm

from config import device
from config import im_size
from utils import get_image

if __name__ == "__main__":
    files = [f for f in os.listdir('video') if f.endswith('.mp4')]
    print('num_files: ' + str(len(files)))

    folder = 'cache'
    if not os.path.isdir(folder):
        os.makedirs(folder)

    print('building index...')
    i = 0
    frames = []
    for file in tqdm(files):
        filename = os.path.join('video', file)
        file = file[3:]
        tokens = file.split('-')
        name = tokens[0] + '-' + tokens[1]

        cap = cv.VideoCapture(filename)
        fps = cap.get(cv.CAP_PROP_FPS)
        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_info = dict()
            frame_info['name'] = name
            frame_info['idx'] = frame_idx
            frame_info['fps'] = fps
            image_fn = os.path.join(folder, str(i) + '.jpg')
            cv.imwrite(image_fn, frame)
            frame_info['image_fn'] = image_fn
            frames.append(frame_info)
            frame_idx += 1
            i += 1

    with open('video_index.json', 'w') as file:
        json.dump(frames, file, ensure_ascii=False, indent=4)

    num_frames = len(frames)
    print('num_frames: ' + str(num_frames))
    assert (i == num_frames)

    checkpoint = 'BEST_checkpoint.tar'
    print('loading model: {}...'.format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model'].module.to(device)
    model.eval()

    print('generating features...')
    with torch.no_grad():
        for frame in tqdm(frames):
            image_fn = frame['image_fn']
            img = cv.imread(image_fn)
            img = cv.resize(img, (im_size, im_size))
            img = get_image(img)
            imgs = torch.zeros([1, 3, im_size, im_size], dtype=torch.float)
            imgs[0] = img
            imgs = imgs.to(device)
            output = model(imgs)
            feature = output[0].cpu().numpy()
            feature = feature / np.linalg.norm(feature)
            frame['feature'] = feature

    with open('video_index.pkl', 'wb') as file:
        pickle.dump(frames, file)
