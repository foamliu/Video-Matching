import time

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config import im_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

checkpoint = 'BEST_checkpoint.tar'
print('loading model: {}...'.format(checkpoint))
checkpoint = torch.load(checkpoint)
model = checkpoint['model'].to(device)
model.eval()
transformer = data_transforms['val']


def get_image(img, transformer):
    img = img[..., ::-1]  # RGB
    img = Image.fromarray(img, 'RGB')  # RGB
    img = transformer(img)
    return img.to(device)


def gen_feature(frame_list):
    file_count = len(frame_list)
    batch_size = 128
    ret_mat = np.empty((file_count, 512), np.float32)

    with torch.no_grad():
        for start_idx in tqdm(range(0, file_count, batch_size)):
            end_idx = min(file_count, start_idx + batch_size)
            length = end_idx - start_idx

            imgs = torch.zeros([length, 3, im_size, im_size], dtype=torch.float)
            for idx in range(0, length):
                i = start_idx + idx
                imgs[idx] = get_image(frame_list[i], transformer)

            features = model(imgs.to(device)).cpu().numpy()
            for idx in range(0, length):
                feature = features[idx]
                feature = feature / np.linalg.norm(feature)
                i = start_idx + idx
                ret_mat[i] = feature

    return ret_mat


if __name__ == "__main__":
    video = 'material/FM190311-10.mp4'
    image = 'material/shancun.JPG'

    cap = cv.VideoCapture(video)

    frame_list = []
    frame_idx = 0

    print('collecting frames...')
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv.resize(frame, (im_size, im_size))
        frame_list.append(frame)
    frame_count = len(frame_list)
    print('frame_count: ' + str(frame_count))

    print('generating features...')
    start = time.time()
    mat = gen_feature(frame_list)
    np.save('video', mat)
    end = time.time()
    elapsed = end - start
    elapsed_per_frame = elapsed / frame_count
    print('elapsed: ' + str(elapsed))
    print('elapsed_per_frame: ' + str(elapsed_per_frame))

    img = cv.imread(image)
    img = cv.resize(img, (im_size, im_size))
    img_list = [img]
    mat = gen_feature(img_list)
    np.save('image', mat[0])
