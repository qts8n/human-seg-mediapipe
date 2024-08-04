import cv2
from numba import njit, prange
import numpy as np
from PIL import Image

_DEFAULT_RESOLUTION = (1920, 1080)
_DEFAULT_NUM_PX = _DEFAULT_RESOLUTION[0] * _DEFAULT_RESOLUTION[1]


def cv2_to_image_a(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))


def image_to_cv2_a(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def cv2_to_image(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def image_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def is_human_present(confidence_mask: np.ndarray, tol: float = 0.2, total: int = _DEFAULT_NUM_PX) -> bool:
    return (confidence_mask.sum() / total) > tol


@njit(fastmath=True, parallel=True, cache=True)
def apply_confidence_mask(image: np.ndarray, confidence_mask: np.ndarray) -> np.ndarray:
    bg_image = image.copy()
    bg_h, bg_w, _ = bg_image.shape
    for h in prange(bg_h):
        for w in prange(bg_w):
            is_human = confidence_mask[h, w]
            if bg_image[h, w, 3] > 0:
                confidence = int(255 * (1 - is_human))
                bg_image[h, w, 3] = confidence
    return bg_image
