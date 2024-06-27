import cv2
from threading import Thread

class VideoStream:
    def __init__(self, input = 0, resolution=(640, 480), framerate=30):
        # Configuro el streaming de cámara USB:
        self.stream = cv2.VideoCapture(0)
        # if not self.stream.isOpened():
        #     raise IOError('Error al obtener imagen del dispositivo')
        
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret = self.stream.set(3, width)
        ret = self.stream.set(4, height)

        # Lee el primer frame del streaming
        (self.grabbed, self.frame) = self.stream.read()

        # Control variable cuando la cámara es detenida
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True
