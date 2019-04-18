import numpy as np
import math
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mat = np.load('video.npy')
    feature = np.load('image.npy')
    print(mat.shape)
    print(feature.shape)
    ret = np.dot(mat, feature)
    print(ret.shape)
    index = np.argmax(ret)
    maximum = ret[index]
    print(index)
    print(maximum)

    threshold = 43.12986168973048
    cosine = np.clip(maximum, -1, 1)
    theta = math.acos(cosine)
    theta = theta * 180 / math.pi
    print(theta)
    print(theta < threshold)

    fps = 25.
    time = 1/fps*index
    print(time)

    plt.plot(ret)
    plt.show()