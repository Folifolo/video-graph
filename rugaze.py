# -*- coding: utf-8 -*
import os
import cv2
import numpy as np
import fnmatch
import random
from abc import ABCMeta, abstractmethod

class SeqGaze(object):
    """
    Взгляд - класс, котрый отавечает за входные данные для сенсоров нейросети.
    Просматривает последовательность входных медиа-данных (картинок или видео,
    в зависимости от реализации)
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def log(self, message):
        """
        Логирование.
        :param message: сообщение для записи его в логи
        :return:
        """
        pass

    @abstractmethod
    def get_shape(self):
        """
        Получить форму объекта взгляда.
        :return:
         (длина, ширина). Например, если взгляд имеет форму 10 на 10 пикселей,
        то вернется (10, 10)
        """
        pass

    @abstractmethod
    def get_next_fixation(self):
        """
        То, что сейчас попало в квадрат взгляда
        :return:
        1)возвращает numpy матрицу значений от 0 до 1.
        Именно это в дальнейшем предназначено подать на вход сенсорам нейросети.
        2) флаг: True если этот кадр первый после "открытия глаза", False иначе
        """
        pass

    @abstractmethod
    def shift(self):
        """
        Установить взгляд в новую произвольную точку на текущих данных.
        :return:
        """

class VideoSeqGaze(SeqGaze):
    """Квадратный взгляд, идущий по видяшкам из заданной папки."""
    def __init__(self, folder_with_videos, side, left_top_coord=None, log=False, show=False):
        assert side > 0
        self.log_enabled = log
        self.frame_shape = None           # щирина и высота кадра у текущего видео
        self.show = show                  # показывать ли то, что оно видит, как видяшку
        self.folder = folder_with_videos  # папка в которой лежат видяшки
        self.side = side                  # сторона квадрата взгляда
        self.prev_frame = None            # предыдущий фрейм видео (не взгляда)
        self.gaze_was_restarted = False   # был ли взгляд только что сдвинут
        self.videos = self._find_all_videos_in_folder(folder_with_videos) # все называния видяшек
        self.left_top_coord = [0,0]       # координата верхнего левого угла взгляда
        self.need_change_gaze_position = True  # нужно ли перестанавливать взгляд в начале каждого видео случайн.образом
        if left_top_coord is not None:  # если она была передана в кнструктор, то на всех видео взгляд будет в ней
            self.need_change_gaze_position = False
            self.left_top_coord = [left_top_coord[0], left_top_coord[1]]
        self.video_generator = self._next_video_name() # генератор, из которого доаставать имя следующего видео
        self.capture = self.open_next_video()          # открываем первое видео

    def log(self, message):
        if self.log_enabled:
            print "<VideoSeqGaze> " + message

    def _find_all_videos_in_folder(self, folder):
        """
        Находит все видео avi и mp4 из данной папки
        :param folder: строка - имя папки
        :return: список строк - имен видяшек из этой папки
        """
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
        """ Получить форму взгляда"""
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
        self.frame_shape = gray.shape
        self.prev_frame = np.zeros(gray.shape)
        self.gaze_was_restarted = True
        if self.need_change_gaze_position:
            self._set_random_left_top()
        return capture

    def close_current_video(self):
        self.capture.release()
        cv2.destroyAllWindows()

    def get_next_fixation(self):
        """
        Возвращает то, что сейчас попало во взгляд (квадратная область текущего кадра видео)
        и является ли это первым кадром с момента переинициализации взгляда
        :return:
        1. numpy матрица чисел от 0 до 1
        2. True, если взгляд был только переинициализирован и False иначе
        """
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
                return None, False # других видео нет, смотреть больше нечего
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subframe = self._get_subframe(self.prev_frame, frame)
        if self.show:
            cv2.imshow('gaze', subframe)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None
        self.prev_frame = frame
        subframe = subframe * float(1) / float(255)
        self.log( str(subframe))
        if self.gaze_was_restarted == True:
            self.gaze_was_restarted = False
            return subframe, True
        else:
            return subframe, False

    def _get_subframe(self, frame1, frame2):
        diff = frame2 - frame1
        X1 = self.left_top_coord[0]
        X2 = self.left_top_coord[0] + self.side
        Y1 = self.left_top_coord[1]
        Y2 = self.left_top_coord[1] + self.side
        return diff[X1:X2, Y1:Y2]

    def _set_random_left_top(self):
        assert self.frame_shape is not None
        max_x = self.frame_shape[0] - self.side
        max_y = self.frame_shape[1] - self.side
        self.left_top_coord[0] = random.randint(0, max_x)
        self.left_top_coord[1] = random.randint(0, max_y)

    def restart(self):
        print "GAZE RESTARTED "
        if self.capture is not None:
             if self.capture.isOpened():
                self.capture.release()
                cv2.destroyAllWindows()
        self.video_generator = self._next_video_name()
        self.capture = self.open_next_video()

    def shift(self, mode='random'):
        if mode == 'random':
            self._set_random_left_top()
            self.gaze_was_restarted = True
        else:
            if mode == 'left':
                self.left_top_coord[0] += 10
                self.left_top_coord[1] += 10


# Бывает, что взгляд установлен в неудачные координаты
# ( т.е. в даннных координатах в кадре ничего не происходит
# и нейросети нечему учиться), тогда взляд надо сдвинуть.
class GazeHistoryAuditor:
    def __init__(self):
        # предыстория из 3 последних фреймов
        self.old = None
        self.now = None
        self.new = None

    def do_we_need_shift(self, new_frame):
        self.old = self.now
        self.now = self.new
        self.new = new_frame
        if self.now is None or \
            self.old is None or \
            self.new is None:
            return False
        result = self.new + self.old + self.now
        if np.sum(result) < 0.5:  # слишком мало активности за последние кадры в этом месте
            return True           # значит нужен шифт
        else:
            return False

    def reset(self):
        # посе того, как взгляд был, например, сдвинут, надо "забыть" предысторию старого взягляда
        self.old = None
        self.now = None
        self.new = None

class GazeTest:
    def __init__(self):
        pass

    def test(self):
        print "seq_video-gaze-test"
        folder = 'dataset'
        gaze = VideoSeqGaze(folder, side=5, log=False, show=True)
        while True:
            img = gaze.get_next_fixation()
            if img is None:
                break


from enum import Enum

class Directions(Enum):
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4

class PicturesSeqGaze(SeqGaze):
    """ Класс квадратного взгляда, смотрящего картинки скользящим окном"""
    def __init__(self):
        pass



if __name__ == "__main__":
    my_gaze = GazeTest()
    my_gaze.test()