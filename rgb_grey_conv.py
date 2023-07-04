import numpy as np


def rgb_to_gray(rgb_image):
    height, width, _ = rgb_image.shape
    gray_image = np.empty((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            r, g, b = rgb_image[i, j]
            # Calculate the grayscale value using the formula: Y = 0.299R + 0.587G + 0.114B
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            gray_image[i, j] = gray_value

    return gray_image
