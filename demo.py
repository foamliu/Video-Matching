import math
import pickle

import numpy as np
import torch
import cv2 as cv
from config import device, pickle_file, im_size
from utils import get_image, get_prob

if __name__ == "__main__":
    with open(pickle_file, 'rb') as file:
        frames = pickle.load(file)

    num_frames = len(frames)
    features = np.empty((num_frames, 512), dtype=np.float32)
    name_list = []
    idx_list = []
    fps_list = []

    for i, frame in enumerate(frames):
        name = frame['name']
        feature = frame['feature']
        fps = frame['fps']
        idx = frame['idx']
        features[i] = feature
        name_list.append(name)
        idx_list.append(idx)
        fps_list.append(fps)

    print(features.shape)
    assert (len(name_list) == num_frames)

    checkpoint = 'BEST_checkpoint.tar'
    print('loading model: {}...'.format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    model = model.to(device)
    model.eval()

    test_fn = 'images/test_img_1.jpg'
    img = cv.imread(test_fn)
    img = cv.resize(img, (im_size, im_size))
    img = get_image(img)
    imgs = torch.zeros([1, 3, im_size, im_size], dtype=torch.float)
    imgs[0] = img
    with torch.no_grad():
        output = model(imgs)
        feature = output[0].cpu().numpy()
        x = feature / np.linalg.norm(feature)

    cosine = np.dot(features, x)
    cosine = np.clip(cosine, -1, 1)
    print('cosine.shape: ' + str(cosine.shape))
    max_index = int(np.argmax(cosine))
    max_value = cosine[max_index]
    print('max_index: ' + str(max_index))
    print('max_value: ' + str(max_value))
    print('name: ' + name_list[max_index])
    print('fps: ' + idx_list[max_index])
    print('idx: ' + idx_list[max_index])
    theta = math.acos(max_value)
    theta = theta * 180 / math.pi

    print('theta: ' + str(theta))
    prob = get_prob(theta)
    print('prob: ' + str(prob))
