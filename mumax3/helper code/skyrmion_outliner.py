import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(input_path, output_path):
    # Read the image
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read the image from {input_path}.")
        return

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(image, (7, 7), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=90)

    # Save the output image
    
    np.save(output_path, edges)
    
    # Optional: Display the images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Edge Detection')
    plt.imshow(edges, cmap='gray')
    plt.show()

# Example usage
input_image_path = r'C:\Users\jsche\Documents\GitHub\sup-spin\mumax3\Stray field resources\stray_field_v8_FeCo_2_6.out\m_75mT_10K_FeCo_2_6_layer1.tif'
output_image_path = r'C:\Users\jsche\Documents\GitHub\sup-spin\mumax3\Stray field resources\stray_field_v8_FeCo_2_6.out\m_75mT_10K_FeCo_2_6_layer1_outline'
process_image(input_image_path, output_image_path)
