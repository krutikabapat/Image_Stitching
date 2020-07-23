# Image_Stitching
Create Image Panorama using OpenCV ( Image Stitching)  

For Image stitching, the following steps have to be followed:-  

1. Read the two images, which have to be joined.  
2. Compute the sift-keypoints and decriptors for both the images.  
3. Computer the inter-descriptor distance between the Images.  
4. Select the top ‘m’ matches for each descriptor of an image.  
5. Run RANSAC to estimate homography.  
6. Warp the to align sticthing.  
7. Finally, stitch them together.      

## References:    

1. https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/    
2. https://towardsdatascience.com/image-stitching-using-opencv-817779c86a83    





