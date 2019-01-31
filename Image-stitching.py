import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randrange

img_ = cv2.imread('/home/krutika/Documents/Image_Stitching/newspaper1.jpg')
img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)

img = cv2.imread('/home/krutika/Documents/Image_Stitching/newspaper2.jpg')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2) 


#print matches
# Apply ratio test
good = []
for m in matches:
     if m[0].distance < 0.5*m[1].distance:         
     	good.append(m)
matches = np.asarray(good)
print(matches.shape)
'''print matches[2,0].queryIdx
print matches[2,0].trainIdx
print matches[2,0].distance'''


if len(matches[:,0]) >= 10:
    src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)

    H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    
else:
    raise AssertionError("Can't find enough keypoints.")  	
   
dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))     	
dst[0:img.shape[0], 0:img.shape[1]] = img
plt.figure()
plt.tight_layout()
plt.subplot(121),plt.imshow(img1),plt.title('First Image')
plt.subplot(122),plt.imshow(img2),plt.title('Second Image')
# plt.imshow(dst)
plt.show()

cv2.imwrite('resultant_stitched_panorama.jpg',dst)
cv2.imshow("img1", img1)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()