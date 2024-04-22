import pywt
#omar samy 20200758,yara ahmed 20201219

import numpy as np
import cv2
import matplotlib.pyplot as plt

def reconstruct_image(cA, cH, cV, cD):
    """Reconstruct the original image using inverse wavelet transform"""
    coeffs = (cA, (cH, cV, cD))
    reconstructed_image = pywt.idwt2(coeffs, 'db1')  # Use the same wavelet as before
    return reconstructed_image

# Load image 
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply 2D wavelet transform
coeffs = pywt.dwt2(image, 'db1')

# Extract four subbands
cA, (cH, cV, cD) = coeffs

 
# Create a 2x2 grid for subplots
fig, axs = plt.subplots(2, 2, figsize=(6, 6))

# Display each image in a subplot
axs[0, 0].imshow(cA, cmap='gray')
axs[0, 0].set_title("approximation")
axs[0, 1].imshow(cH, cmap='gray')
axs[0, 1].set_title("horizontal")
axs[1, 0].imshow(cV, cmap='gray')
axs[1, 0].set_title("vertical")
axs[1, 1].imshow(cD, cmap='gray')
axs[1, 1].set_title("diagonal")

# display the figure
plt.tight_layout()
plt.show()

# display the original image
original_image=reconstruct_image(cA, cH, cV, cD)
plt.imshow(original_image, cmap='gray') 
plt.title("reconstruct image")
plt.show()
