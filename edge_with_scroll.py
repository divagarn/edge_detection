import numpy as np
import cv2
from k_gauss import apply_gaussian_filter
from rgb_grey_conv import rgb_to_gray

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])


def update_image(_=None):
    sigma = cv2.getTrackbarPos("Gauss_Sigma", "Image")
    kernel_size = cv2.getTrackbarPos("Gauss_Kernel_Size", "Image")
    kernel_size = max(1, kernel_size)
    print(kernel_size)
    low_threshold = cv2.getTrackbarPos("Low Threshold", "Image")
    high_threshold = cv2.getTrackbarPos("High Threshold", "Image")

    image_ = np.copy(image_original)
    image = apply_gaussian_filter(image_, sigma, kernel_size)

    gradient_x = np.zeros_like(image, dtype=np.float32)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            gradient_x[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * sobel_x)

    gradient_y = np.zeros_like(image, dtype=np.float32)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            gradient_y[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * sobel_y)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = gradient_magnitude.astype(np.uint8)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    gradient_direction = gradient_direction * 180 / np.pi
    gradient_direction[gradient_direction < 0] += 180

    suppressed = np.zeros_like(gradient_magnitude)
    for i in range(1, image_original.shape[0] - 1):
        for j in range(1, image_original.shape[1] - 1):
            current_mag = gradient_magnitude[i, j]
            current_dir = gradient_direction[i, j]

            if (0 <= current_dir < 22.5 or 157.5 <= current_dir <= 180) and \
               (current_mag >= gradient_magnitude[i, j - 1]) and \
               (current_mag >= gradient_magnitude[i, j + 1]):
                suppressed[i, j] = current_mag
            elif (22.5 <= current_dir < 67.5) and \
                 (current_mag >= gradient_magnitude[i - 1, j - 1]) and \
                 (current_mag >= gradient_magnitude[i + 1, j + 1]):
                suppressed[i, j] = current_mag
            elif (67.5 <= current_dir < 112.5) and \
                 (current_mag >= gradient_magnitude[i - 1, j]) and \
                 (current_mag >= gradient_magnitude[i + 1, j]):
                suppressed[i, j] = current_mag
            elif (112.5 <= current_dir < 157.5) and \
                 (current_mag >= gradient_magnitude[i - 1, j + 1]) and \
                 (current_mag >= gradient_magnitude[i + 1, j - 1]):
                suppressed[i, j] = current_mag

    edges = np.zeros_like(suppressed)
    edges[(suppressed >= high_threshold)] = 255
    edges[(suppressed <= low_threshold)] = 0

    for i in range(1, image_original.shape[0] - 1):
        for j in range(1, image_original.shape[1] - 1):
            if low_threshold <= suppressed[i, j] <= high_threshold:
                if np.max(edges[i-1:i+2, j-1:j+2]) == 255:
                    edges[i, j] = 255

    cv2.imshow("image_color", image_current)
    # cv2.imshow("image_grey", image_original)
    # cv2.imshow("image_gauss", image)
    # cv2.imshow("filtered_x", gradient_x)
    # cv2.imshow("filtered_y", gradient_y)
    # cv2.imshow("filtered_magnitude", gradient_magnitude)
    # cv2.imshow("suppressed_image", suppressed)
    cv2.imshow("images_edge", edges)


image_current = cv2.imread("../filters/color.jpeg")
image_original = rgb_to_gray(image_current)

cv2.namedWindow("Image")
cv2.createTrackbar("Gauss_Sigma", "Image", 1, 10, update_image)
cv2.createTrackbar("Gauss_Kernel_Size", "Image", 5, 10, update_image)
cv2.createTrackbar("Low Threshold", "Image", 25, 100, update_image)
cv2.createTrackbar("High Threshold", "Image", 76, 100, update_image)

update_image()
cv2.waitKey(0)
cv2.destroyAllWindows()
