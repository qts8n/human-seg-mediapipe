import cv2
import mediapipe as mp
import numpy as np

from animation import AnimationState, Animation
from inference import Segmenter

_WINDOW_NAME = 'frame'

_MODEL_PATH = 'assets/selfie_segmenter_landscape.tflite'

_FOREGROUND_PATH = 'assets/foreground.png'

_BACKGROUND_IN_DIR = 'assets/in'
_BACKGROUND_OUT_DIR = 'assets/out'
_BACKGROUND_IDLE_DIR = 'assets/idle'


def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found: {bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found: {fg_channels}'

    # center by default
    if x_offset is None:
        x_offset = (bg_w - fg_w) // 2
    if y_offset is None:
        y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1:
        return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite


def is_human_present(confidence_mask: np.ndarray) -> bool:
    return confidence_mask.sum() > 4000


def _main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
        exit()

    state = AnimationState.ABSENT
    segmenter = Segmenter(_MODEL_PATH)

    foreground_image = cv2.imread(_FOREGROUND_PATH, cv2.IMREAD_UNCHANGED)
    background_in = Animation(_BACKGROUND_IN_DIR)
    background_out = Animation(_BACKGROUND_OUT_DIR)
    background_idle = Animation(_BACKGROUND_IDLE_DIR)

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print('Can\'t receive frame (stream end?). Exiting ...')
            break

        # Our operations on the frame come here
        frame = cv2.resize(frame, (1920, 1080))
        segmenter.segment_async(frame)

        confidence_mask = segmenter.confidence_mask
        if not isinstance(confidence_mask, np.ndarray):
            continue

        human_mask = confidence_mask > 0.1
        human_present = is_human_present(confidence_mask)

        background_image = None

        if state is AnimationState.IN:
            background_image = background_in.next_frame()
            if background_image is None:  # animation ended
                state = AnimationState.IDLE
        elif state is AnimationState.OUT:
            background_image = background_out.next_frame()
            if background_image is None:  # animation ended
                state = AnimationState.ABSENT

        if state is AnimationState.IDLE:
            background_image = background_idle.next_frame(cycle=True)
            if not human_present:
                state = AnimationState.OUT
        elif state is AnimationState.ABSENT:
            if human_present:
                state = AnimationState.IN

        if background_image is not None:
            bg_image = background_image.copy()
            bg_image[human_mask, 3] = 0

            add_transparent_image(frame, bg_image)

        add_transparent_image(frame, foreground_image, y_offset=200)

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
