# -*- coding: utf-8 -*

import random
import os

import itertools
import networkx as nx
import matplotlib.pyplot as plt
import cv2
import numpy as np

import utils

id_generator = itertools.count(0)
name1 = 'bigvideo.mp4'
name2 = 'sample.avi'


# Сеть шарит взглядом по бегущему видеопотоку
# Взгляд ставится там, где что-то происходит (большой дифф между соседними кадрами)
# При прочих равных - рядом с предыдущей фиксацией
# Сеть может сама контролировать взгляд, если захочет, через метод shift
# Если в этой окрестности кончилась активность, то история сдвигов обнуляется
class RuGaze:
    def __init__(self, side):
        self.center = None
        self.side = side
        self.trajectory = []

    def showGaze(self, img):
        utils.draw_anchor(self.center[0], self.center[1], img)
        utils.draw_quad(self.center[0], self.center[1], self.side, img)

    def updateGaze(self, img):
        if len(self.trajectory) == 0:
            self.center, self.side = self._findNewPlaceForGaze(img)
        else:
            pass #TODO find nearest activity?

    def _findNewPlaceForGaze(self, img):
        points = self._findKeyPoints(img)
        num = random.randint(0, len(points)- 1)
        point = points[num]
        newCenter =  (int(point.pt[0]), int(point.pt[1]))
        newSide = int(point.size)*10 #TODO fix magic 10
        return newCenter, newSide


    def _findKeyPoints(self, img):
        detector = cv2.AgastFeatureDetector_create()
        points = detector.detect(img)
        return points

    def reportLastShift(self):
        pass

    def shift(self, xShift, yShift):
        if self.center is not None:
            self.center[0] += xShift
            self.center[1] += yShift

    def getVisibleArea(self):
        pass

    def isNewTrajectory(self):
        return len(self.trajectory) == 1


class RuGraph:
    def getNodeNeighborhood(self, node, radius): # в топологичесткой окрестности узла ( но не выходя за кластер, если он есть?)
        pass



# Глобальная цель алгоритма - стабилизовать как можно больше
class Learning:
    def __init__(self, video):
        self.video = video
        self.gaze = RuGaze(10)

    def processVideo(self):
        if not os.path.isfile(self.video):
            print 'file doesn\'t exist'
            return

        cap = cv2.VideoCapture(self.video)
        prevFrame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if ret != True:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prevFrame is None:
                prevFrame = frame
                continue
            else:
                diff = utils.pixelwise_diff(prevFrame, frame, 10)  # TODO threshold
                self.gaze.updateGaze(diff)
                self.gaze.showGaze(diff)
                prevFrame = frame

            cv2.imshow('diff', diff)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


#learning = Learning(name1)
#learning.processVideo()
#utils.sample_graph_networkx()

n = 50 # n nodes
p = 0.1


import scipy
def generate_sparse_matrix(n, m):
    M = scipy.sparse.random(n, m, density = 0.25 )
    return M


def add_first_layer(matrix):
    G = nx.Graph()

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            index = id_generator.next()
            G.add_node(index, pos = (i,j), color = matrix[i,j], input = True)


    positions = nx.get_node_attributes(G, 'pos').values()
    colors = nx.get_node_attributes(G, 'color').values()

    plt.set_cmap(plt.cm.get_cmap('Blues'))
    input_nodes = nx.get_node_attributes(G, 'input').keys()
    nx.draw_networkx_nodes(G, nodelist=input_nodes, pos = positions, node_color= colors)

    return G

def matrix_to_neuron(G, matrix):
    index = id_generator.next()
    G.add_node(index)
    nx.draw_networkx_nodes(G, nodelist=index)


#G = add_first_layer (M.A)
#plt.show()

def index_in_matrix(matrix, index):
    if index in np.ndindex(matrix.shape):
        return True
    return False


# итератор по области матрицы
def indexes_submatrix(matrix, center_of_area, shape_of_area):
    X, Y = center_of_area[0], center_of_area[1]
    side_x, side_y = int(shape_of_area[0]/2), int(shape_of_area[1]/2)
    max_x = X + side_x
    min_x = X - side_x
    min_y = Y - side_y
    max_y = Y + side_y

    for x in range(min_x, max_x):
        for y in range (min_y, max_y):
            if index_in_matrix(matrix, (x, y)):
                yield x, y



# while True:
#     try:
#         print points_in_area.next()
#     except StopIteration:
#         break

# получить новый кадр
#   очередная фиксация взгляда - если там есть мгновенная активность, то дальше. Если нет - сменить место фиксации
#       взять область интереса в качестве входного тензора
#           сделать фидфорвард пасс до наивысших стабильных нейронов (синих)
#           смотрим на уровень узнавания паттернов самыми высокими из стабильных нейронов
#           если область интереса удалось выразиь в активациях стабильных нейронов (не было неузнанных низкоуровневых паттернов)
#               то превращаем эту активацию в воспоминание о событии(добавлеяем зеленые нейроны/соединения)
#           если выстретились не узнанные никакими нейронами патерны (коэфф-т ниже порога), то добавляем их ввиде нестабильных нейронов (красных)
#               рядом с теми, кто дал максимальный уровень узнавания этого паттерна
#          выставляем всем новым нейронам возраст и яркость, и запрашиваем обновление области интереса
#          самое яркое из воспиминаний прошлого кадра объявляем объектом-внимания
#          проделюваем с областью интереса то же самое что и в прошлый раз (фидфорвард пасс и добавление нейронов)
#          если есть достаточно яркое события, то ищем события из прошлого раза рядом с ним, и создаем делаем точкой траектории
#
