import numpy as np
import math
import matplotlib.pyplot as plt


def compute_n(N):
    return int(math.log2(N))

def prepare_kpq_table(N):
    n = compute_n(N)
    kpq_table = []
    for k in range(N-1):
        p = int(math.log2(k+1))
        q = k - 2**p + 1
        if p == 0:
            q = min(q, 1)
        else:
            q = min(q, 2**p)
        kpq_table.append((k, p, q))
    return kpq_table

def haar_matrix(N):
    # Step 1: Get the dimensions of the original image N
    # Step 2: Compute n as N = 2^n
    n = compute_n(N)
    
    # Step 3: Prepare the (k,p,q) table
    kpq_table = prepare_kpq_table(N)
    
    # Step 4: Fill the first row of the matrix
    H = np.zeros((N, N))
    for v in range(N):
        H[0, v] = 1 / math.sqrt(N)
    
    # Step 5: Loop to fill the Haar transformation matrix
    for k, p, q in kpq_table:
        for v in range(N):
            z = (v + 0.5) / N
            if ((z >= (q-1) / (2**p)) and (z < (q-0.5) / (2**p))):
                H[k+1, v] = 2**(p/2)
            elif ((z >= (q-0.5) / (2**p)) and (z < q / (2**p))):
                H[k+1, v] = -2**(p/2)
            else:
                H[k+1, v] = 0
    
    return H

def subband_coding(image, HaarMatrix, threshold=None):
    
    # Step 3: Multiply the image by the Haar transform matrix to obtain the subband coefficients
    coefficients = np.dot(HaarMatrix, image)
    
    # Step 4: Threshold the coefficients if necessary
    if threshold is not None:
        coefficients[np.abs(coefficients) < threshold] = 0
    
    return coefficients

def compute_four_subbands(image, HaarMatrix, threshold=None):

    # Perform subband coding
    coefficients = subband_coding(image, HaarMatrix, threshold)
    
    # Split coefficients into four subbands
    height, width = image.shape
    LL = coefficients[:height//2, :width//2]
    LH = coefficients[:height//2, width//2:]
    HL = coefficients[height//2:, :width//2]
    HH = coefficients[height//2:, width//2:]
    
    return coefficients, LL, LH, HL, HH

def inverse_subband_coding(coefficients, HaarMatrix):
    # Step 5: Reconstruct the image using the inverse Haar transform
    return np.dot(HaarMatrix.T, coefficients)


# Load an example image
image = plt.imread("image.jpg")  

# Generate the Haar transform matrix for the image size
N = min(image.shape)
HaarMatrix = haar_matrix(N)
print(HaarMatrix)

# Compute four subbands
coefficients,LL, LH, HL, HH = compute_four_subbands(image, HaarMatrix)

# Reconstruct the image
reconstructed_image = inverse_subband_coding(coefficients, HaarMatrix)

# Visualize the four subbands
plt.figure(figsize=(10, 10))
plt.subplot(3, 3, 1)
plt.imshow(LL, cmap='gray')
plt.title('LL (Approximation)')
plt.axis('off')

plt.subplot(3, 3, 2)
plt.imshow(LH, cmap='gray')
plt.title('LH (Horizontal Detail)')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(HL, cmap='gray')
plt.title('HL (Vertical Detail)')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(HH, cmap='gray')
plt.title('HH (Diagonal Detail)')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')

plt.tight_layout()
plt.show()
