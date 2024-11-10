import numpy as np


def median_filter(image, size=3):
    """Applies a median filter to an image.

    Args:
        image (np.array): The image to be filtered.
        size (int): The size of the kernel.

    Returns:
        np.array: The filtered image.
    """

    mean = np.average(image)
