import cv2 
import matplotlib.pyplot as plt 
from scipy.ndimage import gaussian_filter

p=3
pyramid=[]

def lowpass_gaussian(image, sigma):
    """Apply a lowpass Gaussian filter to the input image."""
    filtered_image = gaussian_filter(image, sigma=sigma)
    return filtered_image

def down_sample(image):
    """downsample image to half size"""

    image = cv2.pyrDown(image)
    return image

def construct_pyramid(image):
    for i in range(1,p):
        filterd=lowpass_gaussian(image,1.0)
        down =down_sample(filterd)
        pyramid.append(down)
        image=down

def upsample(image):
    upsampled_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return upsampled_image

#load imag
img = cv2.imread("image.png") 

#construct image pyramid
pyramid.append(img)
construct_pyramid(img)

#disply pyamid
fig, ax = plt.subplots(nrows=1, ncols=p,dpi=80, figsize=(15, 7),sharex=True, sharey=True)

for i in range(p-1,-1,-1):
    ax[i].imshow(pyramid[i])

plt.show()

#bilinear interpolation for last image and display it
upsampled_image=upsample(pyramid[p-1])
cv2.imshow('Upsampled Image', upsampled_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




