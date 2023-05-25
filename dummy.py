import numpy as np
import cv2

# Create black images for the plots
image1 = np.zeros((400, 400, 3), np.uint8)
image2 = np.zeros((400, 400, 3), np.uint8)

# Generate some random data for plotting
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Scale and shift the data to fit the image size
x = (x * 30).astype(int) + 50
y1 = (y1 * 100).astype(int) + 200
y2 = (y2 * 100).astype(int) + 200

# Draw the plots on the respective images
for i in range(len(x) - 1):
    cv2.line(image1, (x[i], y1[i]), (x[i+1], y1[i+1]), (0, 255, 0), 2)
    cv2.line(image2, (x[i], y2[i]), (x[i+1], y2[i+1]), (0, 0, 255), 2)

# Combine the images horizontally
combined_image = cv2.hconcat([image1, image2])

# Display the combined image
cv2.imshow("Plots", combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()