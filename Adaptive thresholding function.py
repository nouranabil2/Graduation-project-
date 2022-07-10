import cv2
import numpy as np


def thresh(img, window, c):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    jump = window - 2
    #     result=np.zeros((img.shape[0],img.shape[1])
    result = np.array(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            neighbours = []
            for m in range(i - jump, i + jump + 1):
                for n in range(j - jump, j + jump + 1):
                    if (m < img.shape[0] and m >= 0 and n < img.shape[1] and n >= 0):
                        neighbours.append(img[m][n])

            maxi = int(max(neighbours))
            mini = int(min(neighbours))

            T = int(((maxi + mini) / 2) - c)
            if (img[i][j] > T):
                result[i][j] = 255
            else:
                result[i][j] = 0

    return result
