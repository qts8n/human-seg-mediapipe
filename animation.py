import enum
import os

import cv2
from PIL import Image


class AnimationState(enum.Enum):
    IDLE = 0
    IN = 1
    OUT = 2
    ABSENT = 3
    DELAY = 4


class Animation:
    def __init__(self, frame_dir: str, resolution: tuple[int, int], pil=False):
        assert os.path.isdir(frame_dir), 'invalid frame dir'

        self.frames = []
        for frame_name in sorted(os.listdir(frame_dir)):
            frame_path = os.path.join(frame_dir, frame_name)
            if not pil:
                frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                frame = cv2.resize(frame, resolution)
            else:
                frame = Image.open(frame_path)
                frame = frame.resize(resolution)
            self.frames.append(frame)

        assert self.frames, 'frame dir is empty'

        self.frame_index = 0

    def next_frame(self, cycle: bool = False):
        try:
            frame = self.frames[self.frame_index]
            self.frame_index += 1
            return frame
        except IndexError:
            self.frame_index = 0
            if cycle:
                return self.frames[self.frame_index]
            return None
