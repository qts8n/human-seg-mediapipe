import time

import mediapipe as mp
import numpy as np


class Segmenter:
    def __init__(self, model_path: str):
        self.result = None

        # callback function
        def update_result(result, *_):
            self.result = result

        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            result_callback=update_result,
        )

        # initialize segmenter
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)

    def segment_async(self, frame: np.ndarray):
        # convert np frame to mp image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # detect landmarks
        self.segmenter.segment_async(mp_image, int(time.time() * 1000))

    @property
    def category_mask(self):
        if self.result is None:
            return None
        return self.result.category_mask.numpy_view()

    @property
    def confidence_mask(self):
        if self.result is None:
            return None
        return self.result.confidence_masks[0].numpy_view()

    def close(self):
        # close segmenter
        self.segmenter.close()
