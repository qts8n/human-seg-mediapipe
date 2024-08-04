# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from time import perf_counter

import cv2
from numba import njit, prange
import numpy as np
from PIL import Image

from animation import AnimationState, Animation, CompositeAnimation
from inference import Segmenter

_WINDOW_NAME = 'frame'

_MODEL_PATH = 'assets/selfie_segmenter_landscape.tflite'

_FOREGROUND_ANIMATION_DIR = 'assets/front_up'
_BACKGROUND_ANIMATION_DIR = 'assets/back_up'
_ANIMATION_DELAY = 60

_RESOLUTION = (1920, 1080)
_NUM_PX = _RESOLUTION[0] * _RESOLUTION[1]

_HUMAN_PRESENCE_TOL = 0.05
_HUMAN_PRESENCE_DELAY = 5
_HUMAN_ABSENCE_DELAY = 5


def cv2_to_image_a(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))


def image_to_cv2_a(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)


def cv2_to_image(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def image_to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def is_human_present(confidence_mask: np.ndarray, tol: float = 0.2) -> bool:
    return (confidence_mask.sum() / _NUM_PX) > tol


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


def _main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
        exit()

    segmenter = Segmenter(_MODEL_PATH)

    foreground_animation = Animation(_FOREGROUND_ANIMATION_DIR, _RESOLUTION, pil=True, offset_out=_ANIMATION_DELAY)
    background_animation = Animation(_BACKGROUND_ANIMATION_DIR, _RESOLUTION, offset_in=_ANIMATION_DELAY)
    animation = CompositeAnimation(foreground_animation, background_animation)

    human_present = False
    human_presence = 0
    human_absence = 0

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        t1_start = perf_counter()
        # if frame is read correctly ret is True
        if not ret:
            print('Can\'t receive frame (stream end?). Exiting ...')
            break

        # Our operations on the frame come here
        frame = cv2.resize(frame, _RESOLUTION)
        segmenter.segment_async(frame)

        confidence_mask = segmenter.confidence_mask
        if not isinstance(confidence_mask, np.ndarray):
            continue

        if is_human_present(confidence_mask, tol=_HUMAN_PRESENCE_TOL):
            human_absence = 0
            if not human_present:
                human_presence += 1
                if human_presence > _HUMAN_PRESENCE_DELAY:
                    human_present = True
        else:
            human_presence = 0
            if human_present:
                human_absence += 1
                if human_absence > _HUMAN_ABSENCE_DELAY:
                    human_present = False

        foreground_image, background_image = animation.current_frames(present=human_present)

        frame = cv2_to_image(frame)

        bg_image = cv2_to_image_a(apply_confidence_mask(background_image, confidence_mask))
        frame.paste(bg_image, mask=bg_image)
        frame.paste(foreground_image, mask=foreground_image)

        frame = image_to_cv2(frame)

        cv2.imshow(_WINDOW_NAME, frame)

        key_pressed = cv2.waitKey(1)
        if key_pressed == ord('q'):
            break
        elif key_pressed == ord('f'):
            cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key_pressed == ord('n'):
            cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        t1_stop = perf_counter()

        print(1/(t1_stop - t1_start), 'fps')

    cv2.destroyAllWindows()
    segmenter.close()


if __name__ == '__main__':
    _main()
