import argparse

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

_Q_BTN = (ord('q'), 202, 233)
_F_BTN = (ord('f'), 193, 224)
_N_BTN = (ord('n'), 212, 242)


def _main(capture: ThreadedCamera, segmenter: Segmenter):
    foreground_animation = Animation(_FOREGROUND_ANIMATION_DIR, _RESOLUTION, pil=True, offset_out=_ANIMATION_DELAY)
    background_animation = Animation(_BACKGROUND_ANIMATION_DIR, _RESOLUTION, offset_in=_ANIMATION_DELAY)
    animation = CompositeAnimation(foreground_animation, background_animation)

    human_present = False
    human_presence = 0
    human_absence = 0

    frame_time_limit = int(1. / _FRAME_RATE * 1000)

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WINDOW_NAME, *_WINDOW_RESOLUTION)

    while True:
        # if frame is read correctly ret is True
        if not capture.status:
            print('Can\'t receive frame (stream end?). Exiting ...')
            break

        frame = capture.frame
        if frame is None:
            continue

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

        key_pressed = cv2.waitKey(frame_time_limit)
        if key_pressed in _Q_BTN:
            break
        elif key_pressed in _F_BTN:
            cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key_pressed in _N_BTN:
            cv2.setWindowProperty(_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Human segmentation filter')
    parser.add_argument(
        '-c', '--camera',
        type=int,
        default=0,
        help='Camera index')
    args = parser.parse_args()

    capture = ThreadedCamera(args.camera)
    segmenter = Segmenter(_MODEL_PATH)
    try:
        _main(capture, segmenter)
    finally:
        capture.stop()
        capture.join()
        cv2.destroyAllWindows()
        segmenter.close()
