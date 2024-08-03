# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from time import perf_counter

import cv2
from numba import njit, prange
import numpy as np
from PIL import Image

from animation import AnimationState, Animation
from inference import Segmenter

_WINDOW_NAME = 'frame'

_MODEL_PATH = 'assets/selfie_segmenter_landscape.tflite'

_FOREGROUND_IN_DIR = 'assets/front_up'
_FOREGROUND_OUT_DIR = 'assets/front_down'
_FOREGROUND_IDLE_DIR = 'assets/front_idle'

_BACKGROUND_IN_DIR = 'assets/back_up'
_BACKGROUND_OUT_DIR = 'assets/back_down'
_BACKGROUND_IDLE_DIR = 'assets/back_idle'

_RESOLUTION = (1920, 1080)
_NUM_PX = _RESOLUTION[0] * _RESOLUTION[1]

_DELAY = 60

_HUMAN_PRESENCE_TOL = 0.05
_HUMAN_PRESENCE_DELAY = 15
_HUMAN_ABSENCE_DELAY = 15


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

    fg_state = AnimationState.ABSENT

    foreground_in = Animation(_FOREGROUND_IN_DIR, _RESOLUTION, pil=True)
    foreground_out = Animation(_FOREGROUND_OUT_DIR, _RESOLUTION, pil=True)
    foreground_idle = Animation(_FOREGROUND_IDLE_DIR, _RESOLUTION, pil=True)

    fg_delay = 0

    bg_state = AnimationState.ABSENT

    background_in = Animation(_BACKGROUND_IN_DIR, _RESOLUTION)
    background_out = Animation(_BACKGROUND_OUT_DIR, _RESOLUTION)
    background_idle = Animation(_BACKGROUND_IDLE_DIR, _RESOLUTION)

    bg_delay = 0

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

        foreground_image = None

        if fg_state is AnimationState.IN:
            foreground_image = foreground_in.next_frame()
            if foreground_image is None:  # animation ended
                fg_state = AnimationState.IDLE
        elif fg_state is AnimationState.OUT:
            foreground_image = foreground_out.next_frame()
            if foreground_image is None:  # animation ended
                fg_state = AnimationState.ABSENT

        if fg_state is AnimationState.IDLE:
            foreground_image = foreground_idle.next_frame(cycle=True)
            if not human_present:
                fg_delay += 1
                if fg_delay > _DELAY:
                    fg_state = AnimationState.OUT
            else:
                fg_delay = 0
        elif fg_state is AnimationState.ABSENT:
            if human_present:
                fg_state = AnimationState.IN

        background_image = None

        if bg_state is AnimationState.IN:
            background_image = background_in.next_frame()
            if background_image is None:  # animation ended
                bg_state = AnimationState.IDLE
        elif bg_state is AnimationState.OUT:
            background_image = background_out.next_frame()
            if background_image is None:  # animation ended
                bg_state = AnimationState.ABSENT

        if bg_state is AnimationState.IDLE:
            background_image = background_idle.next_frame(cycle=True)
            if not human_present:
                bg_state = AnimationState.OUT
        elif bg_state is AnimationState.ABSENT:
            if human_present:
                bg_delay += 1
                if bg_delay > _DELAY:
                    bg_state = AnimationState.IN
                    bg_delay = 0
            else:
                bg_delay = 0


        frame = cv2_to_image(frame)

        if background_image is not None:
            bg_image = cv2_to_image_a(apply_confidence_mask(background_image, confidence_mask))
            frame.paste(bg_image, mask=bg_image)

        if foreground_image is not None:
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
