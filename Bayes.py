__author__ = "Amir Mallaei"
__email__ = "amirmallaei@gmail.com"

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from Myfuncs import my_mean, my_cov, my_count

# Read the Original Image
img = np.array(Image.open('Images/1.jpg'))
size_img = img.shape

# Read the Masks
water = np.array(Image.open('Images/water.jpg'))
green = np.array(Image.open('Images/green.jpg'))
urban = np.array(Image.open('Images/urban.jpg'))


# Display sample regions
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)

ax1.title.set_text('Original Image')
ax1.imshow(img)
ax2.title.set_text('Water Sample')
ax2.imshow(water, cmap='gray')
ax3.title.set_text('Green Sample')
ax3.imshow(green, cmap='gray')
ax4.title.set_text('Urban Sample')
ax4.imshow(urban, cmap='gray')
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')
plt.show()


# calculating the mean water Section
mean1 = my_mean(water, img)
cov1 = my_cov(water, img, mean1)
count1 = my_count(water)
inv_cov1 = np.linalg.inv(cov1)

# calculating the mean green Section
mean2 = my_mean(green, img)
cov2 = my_cov(green, img, mean2)
count2 = my_count(green)
inv_cov2 = np.linalg.inv(cov2)

# calculating the mean urban Section
mean3 = my_mean(urban, img)
cov3 = my_cov(urban, img, mean3)
count3 = my_count(urban)
inv_cov3 = np.linalg.inv(cov3)


total_count = count1 + count2 + count3
P1 = count1/total_count
P2 = count2/total_count
P3 = count3/total_count


regions = np.zeros((size_img))

# Calculating The G in Bayessian
for i in range(0, size_img[0]):
    for j in range(0, size_img[1]):
        x = np.zeros((3, 1))
        x[0, 0] = img[i, j, 0]
        x[1, 0] = img[i, j, 1]
        x[2, 0] = img[i, j, 2]
        g1 = np.log(P1) - 0.5 * np.log(np.linalg.det(cov1)) - [0.5 * (x-mean1).T @ inv_cov1 @ (x-mean1)]
        g2 = np.log(P2) - 0.5 * np.log(np.linalg.det(cov2)) - [0.5 * (x-mean2).T @ inv_cov2 @ (x-mean2)]
        g3 = np.log(P3) - 0.5 * np.log(np.linalg.det(cov3)) - [0.5 * (x-mean3).T @ inv_cov3 @ (x-mean3)]

        if g1 > g2 and g1 > g3:
            regions[i, j, 2] = 1
        elif g2 > g3 and g2 > g1:
            regions[i, j, 1] = 1
        else:
            regions[i, j, 0] = 1


# Displaying the Results
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(411)
ax2 = fig.add_subplot(412)
ax3 = fig.add_subplot(413)
ax4 = fig.add_subplot(414)


ax1.title.set_text('Water Section')
ax1.imshow(regions[:, :, 2], cmap='gray')
ax2.title.set_text('Green Section')
ax2.imshow(regions[:, :, 1], cmap='gray')
ax3.title.set_text('Urban Section')
ax3.imshow(regions[:, :, 0], cmap='gray')
ax4.imshow(regions)
ax1.axis('off')
ax2.axis('off')
ax3.axis('off')
ax4.axis('off')
plt.show()
