# -*- coding: utf-8 -*
import os
import cv2
import numpy as np
import fnmatch
import random


class SimpleVideoGaze:
    def __init__(self, video_name, side, left_top_coord=None, show=False, print_it=False):
        assert side != 0, "gaze square is zero"
        self.video_name = video_name
        if not os.path.isfile(self.video_name):
            self.side = 0
            print 'file doesn\'t exist'
            return
        self.left_top_coord = left_top_coord
        self.side = side
        self.show = show
        self.print_it = print_it
        if print_it: # весьма замедляет всё, но зато большую матрицу всю видно
            np.set_printoptions(threshold=np.inf)
        self.init_video()

    def init_video(self):
        self.capture = cv2.VideoCapture(self.video_name)
        self.prev_frame = None
        assert self.capture.isOpened(), "video file corrupted?"
        ret, frame = self.capture.read()
        if ret is True:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print 'one-frame video?'

    def restart(self, video_name=None):
        if video_name is not None:
            self.video_name = video_name
        print "GAZE RESTARTED: " + self.video_name
        if self.capture.isOpened():
            self.capture.release()
            cv2.destroyAllWindows()
        self.init_video()

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
        self.prev_frame = frame
        subframe = subframe * float(1) / float(255)
        if self.print_it:
            print subframe
            print "---"
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
            if ret is not True:
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

    def test1(self):
        print "gaze-test"
        video = 'bigvideo.avi'
        gaze = SimpleVideoGaze(video_name=video, print_it=True, show=True, side=25, left_top_coord=(120,120))
        print gaze.get_shape()
        while True:
            img = gaze.get_next_fixation()
            if img is None:
                break
    def test2(self):
        print "seq_video-gaze-test"
        folder = 'dataset'
        gaze = VideoSeqGaze(folder, side=5, log=False, show=True)
        while True:
            img = gaze.get_next_fixation()
            if img is None:
                break

class VideoSeqGaze:
    def __init__(self, folder_with_videos, side, log=False, show=False):
        assert side > 0
        self.log_enabled = log
        self.show = show
        self.left_top_coord = [0, 0]
        self.prev_frame = None
        self.folder = folder_with_videos
        self.side = side
        self.videos = self._find_all_videos_in_folder(folder_with_videos)
        self.video_generator = self._next_video_name()
        self.capture = self.open_next_video()


    def log(self, message):
        if self.log_enabled:
            print message

    def _find_all_videos_in_folder(self, folder):
        results = []
        for root, dirs, files in os.walk(folder):
            for _file in files:
                if fnmatch.fnmatch(_file, '*.avi'):
                    results.append(os.path.join(root, _file))
                if fnmatch.fnmatch(_file, '*.mp4'):
                    results.append(os.path.join(root, _file))
        self.log(str(results))
        return results

    def get_shape(self):
        return (self.side, self.side)

    def _next_video_name(self):
        for video in self.videos:
            yield video

    def open_next_video(self):
        video = next(self.video_generator, None)
        if video is None:
            return None  # все видео уже показаны
        print "go to next video: " + video
        assert os.path.isfile(video)
        capture = cv2.VideoCapture(video)
        self.prev_frame = None
        assert capture.isOpened(), "video file corrupted?"
        ret, frame = capture.read()
        assert ret is True, 'one-frame video?'
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_frame = np.zeros(gray.shape)
        self.set_left_top(self.prev_frame.shape)
        return capture

    def close_current_video(self):
        self.capture.release()
        cv2.destroyAllWindows()

    def get_next_fixation(self):
        assert self.prev_frame is not None,  "video was not opened"
        if not self.capture.isOpened():
            self.close_current_video()
            self.capture = self.open_next_video()  #с этим видео проблемы, начинаем другое
            if self.capture is None:
                return None  # других видео нет, смотреть больше нечего
        ret, frame = self.capture.read()
        if ret is not True:
            # видео кончилось, начинаем следующее - с черного кадра
            self.close_current_video()
            self.capture = self.open_next_video()
            frame = self.prev_frame  # и прошлом и в нынешнем фрейме одна картинка, чтоб дифф был 0
            if self.capture is None:
                return None # других видео нет, смотреть больше нечего
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subframe = self.get_subframe(self.prev_frame, frame)
        if self.show:
            cv2.imshow('gaze', subframe)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None
        self.prev_frame = frame
        subframe = subframe * float(1) / float(255)
        self.log( str(subframe))
        return subframe

    def get_subframe(self, frame1, frame2):
        diff = frame2 - frame1
        X1 = self.left_top_coord[0]
        X2 = self.left_top_coord[0] + self.side
        Y1 = self.left_top_coord[1]
        Y2 = self.left_top_coord[1] + self.side
        return diff[X1:X2, Y1:Y2]

    def set_left_top(self, shape_of_video_frame):
        max_x = shape_of_video_frame[0] - self.side
        max_y = shape_of_video_frame[1] - self.side
        self.left_top_coord[0] = random.randint(0, max_x)
        self.left_top_coord[1] = random.randint(0, max_y)

my_gaze = GazeTest()
my_gaze.test2()

