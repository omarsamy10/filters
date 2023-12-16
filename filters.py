import numpy as np
import cv2
import matplotlib.pyplot as plt

def mean_filter(image, kernel_size):
    rows, cols = image.shape
    result = np.zeros_like(image, dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            # Extract the neighborhood around the current pixel
            neighborhood = image[max(0, i - kernel_size // 2):min(rows, i + kernel_size // 2 + 1),
                                 max(0, j - kernel_size // 2):min(cols, j + kernel_size // 2 + 1)]
            # Calculate the mean of the neighborhood and set it as the new pixel value
            result[i, j] = np.mean(neighborhood)

    return result

def wiener_filter(image, kernel_size, noise_variance=20):
    # Convert the image to float32 for processing
    image_float = np.float32(image)

    # Estimate the local mean using a filter
    local_mean = cv2.boxFilter(image_float, -1, (kernel_size, kernel_size))

    # Estimate the local variance
    local_variance = cv2.boxFilter(image_float ** 2, -1, (kernel_size, kernel_size)) - local_mean ** 2

    # Calculate the noise variance
    noise_estimate = np.mean(local_variance)

    # Compute the Wiener filter
    wiener_filter = local_mean + np.maximum(0, local_variance - noise_variance) / np.maximum(local_variance, noise_estimate)

    # Clip values to the valid range [0, 255]
    wiener_filter = np.clip(wiener_filter, 0, 255)

    # Convert back to uint8
    wiener_filter = np.uint8(wiener_filter)

    return wiener_filter

# Read an example image (replace this with your image path)
image_path = 'image.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
kernel_size=int(input("enter kernel size"))
# Apply the arithmetic mean filter
filtered_image = mean_filter(image,kernel_size)

# Plot the original and filtered images
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('mean Filtered Image')

plt.show()

# Apply the adaptive Wiener filter
kernel_size=int(input("enter wiener kernel size"))
variance=int(input("enter variance"))
filtered_image = wiener_filter(image,kernel_size,variance)

# Plot the original and filtered images
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Wiener Filtered Image')

plt.show()
