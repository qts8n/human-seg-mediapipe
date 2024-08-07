from time import perf_counter
import os

import cv2
import numpy as np

from animation import Animation, CompositeAnimation
from camera import ThreadedCamera
from inference import Segmenter
import utils

_CAMERA_INDEX = 0

_WINDOW_NAME = 'frame'
_WINDOW_RESOLUTION = (1280, 720)

_MODEL_PATH = 'assets/selfie_segmenter_landscape.tflite'

_FOREGROUND_ANIMATION_DIR = 'assets/front_up'
_BACKGROUND_ANIMATION_DIR = 'assets/back_up'
_ANIMATION_DELAY = 60

_FRAME_RATE = 30
_RESOLUTION = (1920, 1080)
_NUM_PX = _RESOLUTION[0] * _RESOLUTION[1]

_HUMAN_PRESENCE_TOL = 0.05
_HUMAN_PRESENCE_DELAY = 5
_HUMAN_ABSENCE_DELAY = 5


def _main():
    segmenter = Segmenter(_MODEL_PATH)

    foreground_animation = Animation(_FOREGROUND_ANIMATION_DIR, _RESOLUTION, pil=True, offset_out=_ANIMATION_DELAY)
    background_animation = Animation(_BACKGROUND_ANIMATION_DIR, _RESOLUTION, offset_in=_ANIMATION_DELAY)
    animation = CompositeAnimation(foreground_animation, background_animation)

    human_present = False
    human_presence = 0
    human_absence = 0

    prev_time = 0
    frame_time_limit = 1. / _FRAME_RATE

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WINDOW_NAME, *_WINDOW_RESOLUTION)

    cap = ThreadedCamera(_CAMERA_INDEX)
    while True:
        frame = cap.frame
        if frame is None:
            continue

        # if frame is read correctly ret is True
        if not cap.status:
            print('Can\'t receive frame (stream end?). Exiting ...')
            break

        # FPS limit to _FRAME_RATE
        time_start = perf_counter()
        time_elapsed = time_start - prev_time
        if time_elapsed < frame_time_limit:
            continue
        prev_time = time_start

        # Our operations on the frame come here
        frame = cv2.resize(frame, _RESOLUTION)
        segmenter.segment_async(frame)

        confidence_mask = segmenter.confidence_mask
        if confidence_mask is None or frame is None:
            continue

        if utils.is_human_present(confidence_mask, tol=_HUMAN_PRESENCE_TOL, total=_NUM_PX):
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

        fg_image, background_image = animation.current_frames(present=human_present)
        segmenter.background_image = background_image

        frame = segmenter.pil_frame
        if frame is None:
            continue
        frame.paste(fg_image, mask=fg_image)
        frame = utils.image_to_cv2(frame)

        cv2.imshow(_WINDOW_NAME, frame)

        key_pressed = cv2.waitKey(1)
        if key_pressed == ord('q'):
            break
        elif key_pressed == ord('f'):
            cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key_pressed == ord('n'):
            cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        time_end = perf_counter() - time_start
        print('frame processed:', time_end * 1000, 'ms')

    cv2.destroyAllWindows()
    segmenter.close()


if __name__ == '__main__':
    _main()
