import time

import mediapipe as mp  # type: ignore
import numpy as np
from PIL import Image


class Segmenter:
    def __init__(self, model_path: str, output_category_mask=False):
        self.result = None
        self.output_category_mask = output_category_mask
        self.category_mask = None
        self.confidence_mask = None

        # callback function
        def update_result(result, *_):
            self.result = result
            if result is None:
                return

            self.confidence_mask = result.confidence_masks[0].numpy_view()

            if output_category_mask:
                self.category_mask = Image.fromarray(result.category_mask.numpy_view())

        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            output_category_mask=output_category_mask,
            result_callback=update_result,
        )

        # initialize segmenter
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)

    def segment_async(self, frame: np.ndarray):
        # convert np frame to mp image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # detect landmarks
        self.segmenter.segment_async(mp_image, int(time.time() * 1000))

    def close(self):
        # close segmenter
        self.segmenter.close()
