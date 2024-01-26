import numpy as np
from skimage.metrics import structural_similarity
from skimage import io

def mse(img1, img2):

    M = img1.shape[0]
    N = img1.shape[1]

    mse = 0

    for i in range(M):
        for j in range(N):
            mse += (np.float64(img1[i, j]) - np.float64(img2[i, j])) ** 2

    return mse / (M * N)

def psnr(img1, img2):

    M = img1.shape[0]
    N = img1.shape[1]

    mse = 0

    for i in range(M):
        for j in range(N):
            mse += ((np.float64(img1[i, j]) - np.float64(img2[i, j])) ** 2)/ (M * N)

    Pmax = np.max(img1)
    return 10 * np.log10(Pmax ** 2 / mse)


