# -*- coding: utf-8 -*
import os
import cv2
import numpy as np

THRESHOLD_FOR_DIFF = 10


class SimpleVideoGaze:
    def __init__(self, videoname, side, left_top_coord=None, show=False, print_it=False):
        assert side != 0, "gaze square is zero"
        self.videoname = videoname
        if not os.path.isfile(self.videoname):
            print 'file doesn\'t exist'
            return
        self. left_top_coord = left_top_coord
        self.capture = cv2.VideoCapture(self.videoname)
        self.prev_frame = None
        self.show = show
        self.print_it = print_it
        if print_it: #весьма замедляет всё, но зато большую матрицу всю видно
            np.set_printoptions(threshold=np.inf)
        self.side = side
        assert self.capture.isOpened(), "video file corrupted?"
        ret, frame = self.capture.read()
        if ret is True:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print 'one-frame video?'


    def get_shape(self):
        return (self.side, self.side)

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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None
        if self.print_it:
            print subframe
            print "---"
        self.prev_frame = frame

        return subframe

    def get_subframe(self, frame1, frame2):
        diff = frame2 - frame1
        X1 = self.left_top_coord[0]
        X2 = self.left_top_coord[0] + self.side
        Y1 = self.left_top_coord[1]
        Y2 = self.left_top_coord[1] + self.side
        return diff[X1:X2, Y1:Y2]

    def show_video(self):
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret != True:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.prev_frame is None:
                self.prev_frame = frame
                continue
            else:
                diff = self.prev_frame - frame
                self.prev_frame = frame

            cv2.imshow('diff', diff)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.capture.release()
        cv2.destroyAllWindows()


class GazeTest:
    def __init__(self):
        pass

    def test(self):
        print "gaze-test"
        video = 'bigvideo.mp4'
        input = SimpleVideoGaze(videoname=video, print_it=True, show=True, side=200, left_top_coord=(220,220))
        print input.get_shape()
        while True:
            img = input.get_next_fixation()
            if img is None:
                break

#gaze = GazeTest()
#gaze.test()

