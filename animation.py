import os

import cv2
from PIL import Image


class Animation:
    def __init__(
            self,
            frame_dir: str,
            resolution: tuple[int, int],
            pil: bool = False,
            offset_in: int = 0,
            offset_out: int = 0,
        ):
        assert os.path.isdir(frame_dir), 'invalid frame dir'

        self.frame_idx = []
        self.frames = []
        for frame_id, frame_name in enumerate(sorted(os.listdir(frame_dir))):
            frame_path = os.path.join(frame_dir, frame_name)
            if not pil:
                frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                frame = cv2.resize(frame, resolution)
            else:
                frame = Image.open(frame_path)
                frame = frame.resize(resolution)
            self.frame_idx.append(frame_id)
            self.frames.append(frame)

        assert self.frames, 'frame dir is empty'

        offset_in_idx = []
        for _ in range(offset_in):
            offset_in_idx.append(0)

        offset_out_idx = []
        for _ in range(offset_out):
            offset_out_idx.append(-1)

        self.frame_idx = offset_in_idx + self.frame_idx + offset_out_idx

        self.frame_num = len(self.frame_idx)

        self.frame_index = 0


    def next_frame(self):
        if self.frame_index >= self.frame_num:
            return self.frames[-1]
        elif self.frame_index < 0:
            self.frame_index = 0
        frame_id = self.frame_idx[self.frame_index]
        frame = self.frames[frame_id]
        self.frame_index += 1
        return frame

    def prev_frame(self):
        if self.frame_index < 0:
            return self.frames[0]
        elif self.frame_index >= self.frame_num:
            self.frame_index = self.frame_num - 1
        frame_id = self.frame_idx[self.frame_index]
        frame = self.frames[frame_id]
        self.frame_index -= 1
        return frame


class CompositeAnimation:
    def __init__(self, fg_animation: Animation, bg_animation: Animation):
        self.fg = fg_animation
        self.bg = bg_animation

    def current_frames(self, present: bool):
        if present:
            return self.fg.next_frame(), self.bg.next_frame()
        return self.fg.prev_frame(), self.bg.prev_frame()
