import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


def compute_img_mean_std(image_paths):
    """
    computing the mean and std of three channel on the whole dataset,
    first we should normalize the image from 0-255 to 0-1
    """

    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.0

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means, stdevs


if __name__ == "__main__":
    path = glob(os.path.join("../dataset", "*", "*.jpg"))
    norm_mean, norm_std = compute_img_mean_std(path)
