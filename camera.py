import os
from threading import Thread
from time import perf_counter

import cv2


class ThreadedCameraOpenError(Exception):
    pass


class ThreadedCamera:
    def __init__(self, src=0):
        if os.name == 'nt':
            self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.capture = cv2.VideoCapture(src)

        if not self.capture.isOpened():
            raise ThreadedCameraOpenError('Unable to open camera')

        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.status = True
        self.frame = None

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()


    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
