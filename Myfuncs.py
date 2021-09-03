import numpy as np

def my_mean(mask, img):
    mask_size = mask.shape;
    count = 0; jam = np.zeros((3, 1));
    for i in range(0,mask_size[0]):
        for j in range(0,mask_size[1]):
            if mask[i,j] > 0:
                count += 1
                jam[0,0] += img[i,j,0]
                jam[1,0] += img[i,j,1]
                jam[2,0] += img[i,j,2]
    return jam/count


def my_count(mask):
    count = 0;mask_size = mask.shape;
    for i in range(0,mask_size[0]):
        for j in range(0,mask_size[1]):
            if mask[i,j] > 0:
                count += 1
    return count


def my_cov(mask, img, avg):
    mask_size = mask.shape;
    count = 0; jam = np.zeros((3, 3));
    avgg = np.matmul(avg,avg.T)
    for i in range(0,mask_size[0]):
        for j in range(0,mask_size[1]):
            if mask[i,j] > 0:
                count += 1; temp = np.zeros((3,1));
                temp[0,0] = img[i,j,0]
                temp[1,0] = img[i,j,1]
                temp[2,0] = img[i,j,2]                
                jam += np.matmul(temp,temp.T) - avgg
    return jam/count
