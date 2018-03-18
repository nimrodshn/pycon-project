import cv2
import numpy as np
from matplotlib import pyplot as plt

img_name = 'a4.jpg'
img = cv2.imread('pictures/' + img_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img,5)

# threshold
th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
ddw = th

# invert image (background to 0, letters to 1)
th = cv2.bitwise_not(th)

# denoising image.
denoise_kernel = np.ones((3,3),np.uint8)
th = cv2.morphologyEx(th, cv2.MORPH_OPEN, denoise_kernel)

# classifier.
classify_kernel = np.ones((2,60),np.uint8)
th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, classify_kernel)
th = cv2.morphologyEx(th, cv2.MORPH_OPEN, classify_kernel)

# denoise again
denoise_kernel = np.ones((10,1),np.uint8)
th = cv2.morphologyEx(th, cv2.MORPH_OPEN, denoise_kernel)

# fixed false negative
denoise_kernel = np.ones((5,5),np.uint8)
th = cv2.dilate(th,denoise_kernel,iterations = 1)

img = cv2.merge((ddw,ddw,th))

plt.imshow(img,'gray')

img = cv2.merge((th,th,th))
cv2.imwrite("pictures/out/" + img_name,img)
plt.show()
