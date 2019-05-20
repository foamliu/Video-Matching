import math

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

im_size = 112


def get_frame_list():
    video = 'material/FM190311-10.mp4'
    cap = cv.VideoCapture(video)
    frame_list = []
    print('collecting frames...')
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv.resize(frame, (im_size, im_size))
        frame_list.append(frame)
    frame_count = len(frame_list)
    print('frame_count: ' + str(frame_count))
    return frame_list


if __name__ == "__main__":
    mat = np.load('video.npy')
    feature = np.load('image.npy')
    print(mat.shape)
    print(feature)
    print(feature.shape)
    frame_count = mat.shape[0]
    cosine = np.dot(mat, feature)
    cosine = np.clip(cosine, -1, 1)
    print(cosine.shape)
    max_index = np.argmax(cosine)
    max_value = cosine[max_index]
    print(max_index)
    print(max_value)

    threshold = 50
    theta = math.acos(max_value)
    theta = theta * 180 / math.pi
    print(theta)
    print(theta < threshold)

    fps = 25.
    time = 1 / fps * max_index
    print(time)

    x = np.linspace(0, frame_count / fps, frame_count)
    plt.plot(x, cosine)
    plt.show()

    frame_list = get_frame_list()
    matched_frame = frame_list[max_index]
    cv.imwrite('match.jpg', matched_frame)
