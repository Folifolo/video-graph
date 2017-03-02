# -*- coding: utf-8 -*
import os
import cv2

THRESHOLD_FOR_DIFF = 10

class SimpleVideoGaze:
    def __init__(self, videoname, side, center=None, show=False, print_it=True):
        assert side != 0, "gaze square is zero"
        self.videoname = videoname
        if not os.path.isfile(self.videoname):
            print 'file doesn\'t exist'
            return
        self.center = center
        self.capture = cv2.VideoCapture(self.videoname)
        self.prev_frame = None
        self.show = show
        self.print_it = print_it
        assert self.capture.isOpened(), "video file corrupted?"
        ret, frame = self.capture.read()
        if ret is True:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print 'one-frame video?'

    def get_shape(self):
        return self.prev_frame.shape

    def get_next_fixation(self):
        if self.prev_frame is None:
            print "video was not opened"
            return None
        if not self.capture.isOpened():
            self.capture.release()
            cv2.destroyAllWindows()
            return None
        ret, frame = self.capture.read()
        if ret is not True:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subframe = self.get_subframe(self.prev_frame, frame)
        if self.show:
            cv2.imshow('gaze', subframe)
        if self.print_it:
            print subframe
        self.prev_frame = frame
        return subframe

    def get_subframe(self, frame1, frame2):
        return frame2 - frame1

class GazeTest:
    def __init__(self):
        pass
    def test(self):
        print "gaze-test"
        video = 'bigvideo.mp4'
        input = SimpleVideoGaze(videoname=video, print_it=True,show=True, side=10)
        print input.get_shape()
        while True:
            img = input.get_next_fixation()
            if img is None:
                break

gaze = GazeTest()
gaze.test()