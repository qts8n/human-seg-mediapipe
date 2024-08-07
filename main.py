from time import perf_counter
import cv2
import numpy as np

from animation import Animation, CompositeAnimation
from inference import Segmenter
import utils

_WINDOW_NAME = 'frame'
_WINDOW_RESOLUTION = (1280, 270)

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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
        exit()
    # may not work in some cases, hence resize next
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, _RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _RESOLUTION[1])

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
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print('Can\'t receive frame (stream end?). Exiting ...')
            break

        # FPS limit to _FRAME_RATE
        curr_time = perf_counter()
        time_elapsed = curr_time - prev_time
        if time_elapsed < frame_time_limit:
            continue
        prev_time = curr_time

        # Our operations on the frame come here
        frame = cv2.resize(frame, _RESOLUTION)
        segmenter.segment_async(frame)

        frame = segmenter.frame
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

        foreground_image, background_image = animation.current_frames(present=human_present)
        segmenter.foreground_image = foreground_image
        segmenter.background_image = background_image

        cv2.imshow(_WINDOW_NAME, frame)

        key_pressed = cv2.waitKey(1)
        if key_pressed == ord('q'):
            break
        elif key_pressed == ord('f'):
            cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key_pressed == ord('n'):
            cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    cv2.destroyAllWindows()
    segmenter.close()


if __name__ == '__main__':
    _main()
