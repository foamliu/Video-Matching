import numpy as np

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