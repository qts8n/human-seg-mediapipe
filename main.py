import argparse
from time import perf_counter

import cv2

from animation import Animation, CompositeAnimation
from camera import ThreadedCamera
from inference import Segmenter
import utils

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

_Q_BTN = (ord('q'), 202)
_F_BTN = (ord('f'), 193)
_N_BTN = (ord('n'), 212)

def _main(cap: ThreadedCamera, verbose: bool = False):
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
        # if frame is read correctly ret is True
        if not cap.status:
            print('Can\'t receive frame (stream end?). Exiting ...')
            break

        frame = cap.frame
        if frame is None:
            continue

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
        if key_pressed in _Q_BTN:
            break
        elif key_pressed in _F_BTN:
            cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key_pressed in _N_BTN:
            cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        if verbose:
            time_end = perf_counter() - time_start
            print('frame processed:', time_end * 1000, 'ms')

    cv2.destroyAllWindows()
    segmenter.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Human segmentation filter')
    parser.add_argument(
        '-c', '--camera',
        type=int,
        default=0,
        help='Camera index')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output')
    args = parser.parse_args()

    cap = ThreadedCamera(args.camera)
    try:
        _main(cap, verbose=args.verbose)
    finally:
        cap.stop()
        cap.join()
