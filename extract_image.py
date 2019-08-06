import os

import cv2 as cv
from tqdm import tqdm

if __name__ == "__main__":
    files = [f for f in os.listdir('video') if f.endswith('.mp4')]
    print('num_files: ' + str(len(files)))

    folder = 'v_images'
    if not os.path.isdir(folder):
        os.makedirs(folder)

    idx = 0
    for file in tqdm(files):
        filename = os.path.join('video', file)
        print(filename)

        cap = cv.VideoCapture(filename)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            cv.imwrite('v_images/{}.jpg'.format(idx), frame)
            idx = idx + 1

    cap.release()
