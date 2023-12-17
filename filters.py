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
image_path = 'Image.png'
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


# Apply the Sobel filter in both x and y directions
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Calculate the gradient magnitude
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

th1=int(input("enter first of 4 threshold values"))
th2=int(input("enter second of 4 threshold values"))
th3=int(input("enter third of 4 threshold values"))
th4=int(input("enter fourth of 4 threshold values"))

# Define different threshold values
threshold_values = [th1 , th2 , th3 , th4]  # Set different threshold values here
# Lowering the threshold valuesIt means 
# more pixels will be classified as edges 
# Create subplots for displaying images
num_thresholds = len(threshold_values)
fig, axes = plt.subplots(1, num_thresholds, figsize=(15, 5))

# Display images for different threshold values
for i, threshold_value in enumerate(threshold_values):
    thresholded_image = np.zeros_like(gradient_magnitude)
    for r in range(gradient_magnitude.shape[0]):
        for c in range(gradient_magnitude.shape[1]):
            if gradient_magnitude[r, c] > threshold_value:
                thresholded_image[r, c] = 255
            else:
                thresholded_image[r, c] = 0
    axes[i].imshow(thresholded_image, cmap='gray')
    axes[i].set_title(f'Threshold: {threshold_value}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
