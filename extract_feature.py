import cv2 as cv
from models import model
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm
import numpy as np
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

def gen_feature(path):
    print('gen features {}...'.format(path))
    # Preprocess the total files count
    files = []
    for filepath in walkdir(path, '.jpg'):
        files.append(filepath)
    file_count = len(files)

    batch_size = 128

    with torch.no_grad():
        for start_idx in tqdm(range(0, file_count, batch_size)):
            end_idx = min(file_count, start_idx + batch_size)
            length = end_idx - start_idx

            imgs = torch.zeros([length, 3, 112, 112], dtype=torch.float)
            for idx in range(0, length):
                i = start_idx + idx
                filepath = files[i]
                imgs[idx] = get_image(cv.imread(filepath, True), transformer)

            features = model(imgs.to(device)).cpu().numpy()
            for idx in range(0, length):
                i = start_idx + idx
                filepath = files[i]
                tarfile = filepath + '_0.bin'
                feature = features[idx]
                write_feature(tarfile, feature / np.linalg.norm(feature))

def process(frame):
    pass


if __name__ == "__main__":
    video = 'material/FM190311-10.mp4'
    image = 'material/9707366.jpg'

    cap = cv.VideoCapture(video)

    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        process(frame)

        frame_idx = frame_idx + 1
