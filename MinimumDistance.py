__author__ = "Amir Mallaei"
__email__ = "amirmallaei@gmail.com"

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

from Myfuncs import my_mean

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

# calculating the mean green Section
mean2 = my_mean(green, img)

# calculating the mean urban Section
mean3 = my_mean(urban, img)


regions = np.zeros((size_img))

# Calculating The G in Minimal Distance
for i in range(0, size_img[0]):
    for j in range(0, size_img[1]):
        x = img[i, j]
        g1 = np.matmul(x, mean1) - 0.5 * np.matmul(np.transpose(mean1), mean1)
        g2 = np.matmul(x, mean2) - 0.5 * np.matmul(np.transpose(mean2), mean2)
        g3 = np.matmul(x, mean3) - 0.5 * np.matmul(np.transpose(mean3), mean3)
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
