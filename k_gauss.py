import cv2
import numpy as np


def apply_gaussian_filter(image, sigma, kernel_size):
    size = kernel_size
    channels = cv2.split(image)
    filtered_channels = []

    for channel in channels:
        kernel = np.zeros((size, size))
        # print(kernel)
        center = size // 2
        # print(center)

        for i in range(size):
            for j in range(size):
                x = i - center
                y = j - center
                kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

        kernel /= np.sum(kernel)
        # print(kernel)
        # print(np.shape(kernel))

        padded_channel = np.pad(channel, ((center, center), (center, center)), mode='constant')

        filtered_channel = np.zeros_like(channel)

        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                filtered_channel[i, j] = np.sum(padded_channel[i:i + size, j:j + size] * kernel)

        filtered_channels.append(filtered_channel)

    filtered_image = cv2.merge(filtered_channels)
    # print(channel.dtype)

    return filtered_image.astype(image.dtype)


# image = cv2.imread("../filters/color.jpeg")
# sigma = 3
# kernel_size = 5
# filtered_image = apply_gaussian_filter(image, sigma, kernel_size)
# cv2.imshow("Original Image", image)
# cv2.imshow("Filtered Image", filtered_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
