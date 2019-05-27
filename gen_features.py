import os
import pickle

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config import device
from config import im_size


def get_image(img):
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    return img.to(device)


if __name__ == "__main__":
    checkpoint = 'BEST_checkpoint.tar'
    print('loading model: {}...'.format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model'].to(device)
    model.eval()

    data_transforms = {
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    transformer = data_transforms['val']

    folder = 'video'
    files = [f for f in os.listdir(folder) if f.endswith('.mp4')]
    frames = []

    for file in tqdm(files):
        filename = os.path.join(folder, file)
        file = file[3:]
        tokens = file.split('-')
        name = tokens[0] + '-' + tokens[1]
        # print(name)

        cap = cv.VideoCapture(filename)
        fps = cap.get(cv.CAP_PROP_FPS)

        frame_idx = 0
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                frame_info = dict()
                frame_info['name'] = name
                frame_info['idx'] = frame_idx
                frame_info['fps'] = fps

                frame = cv.resize(frame, (im_size, im_size))
                img = get_image(frame)
                imgs = torch.zeros([1, 3, im_size, im_size], dtype=torch.float)
                imgs[0] = img
                with torch.no_grad():
                    output = model(imgs)
                    feature = output[0].cpu().numpy()
                    feature = feature / np.linalg.norm(feature)
                    frame_info['feature'] = feature

                    frame_idx += 1

    with open('video_index.pkl', 'wb') as file:
        pickle.dump(frames, file)

    num_frames = len(frames)
    print('num_frames: ' + str(num_frames))
