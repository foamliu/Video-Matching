import math

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

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
    index = np.argmax(cosine)
    maximum = cosine[index]
    print(index)
    print(maximum)

    threshold = 43.12986168973048
    theta = math.acos(maximum)
    theta = theta * 180 / math.pi
    print(theta)
    print(theta < threshold)

    fps = 25.
    time = 1 / fps * index
    print(time)

    x = np.linspace(0, frame_count / fps, frame_count)
    plt.plot(x, cosine)
    plt.show()

    image = 'material/9707366.jpg'
    img = cv.imread(image)
    img = cv.resize(img, (112, 112))
    cv.imwrite('material/out.jpg', img)
