import numpy as np
import matplotlib.pyplot as plt

import cv2
from skimage.measure import compare_ssim as ssim


img1 = cv2.imread('results/volcano.jpg',0)
img2 = cv2.imread('results/volcano_PGD_Attack.jpg',0)
rows, cols = img1.shape

def mse(x, y):
    return np.linalg.norm(x - y)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
                         sharex=True, sharey=True)
ax = axes.ravel()

mse_none = mse(img1, img2)
ssim_none = ssim(img1, img2)


label = 'MSE: {:.6f}, SSIM: {:.6f}'

ax[0].imshow(img1)
ax[0].set_title('Original image')

ax[1].imshow(img2)
ax[1].set_xlabel(label.format(mse_none, ssim_none))
ax[1].set_title('Image with noise')

plt.tight_layout()
plt.show()