# -*- coding: utf-8 -*

import random
import os
import networkx as nx
import matplotlib.pyplot as plt
import cv2
import numpy as np

import utils


name1 = 'bigvideo.mp4'
name2 = 'sample.avi'
print 'opencv version: ' + cv2.__version__

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
    M = scipy.sparse.random(n, m, density = 0.65 )
    return M


def add_first_layer(matrix):
    G = nx.Graph()
    index = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            G.add_node(index, pos = (i,j), color = matrix[i,j])
            index +=1

    coordinates = nx.get_node_attributes(G, 'pos').values()
    colors = nx.get_node_attributes(G, 'color').values()
    nx.draw(G, pos = coordinates, node_color= colors)
    plt.show()



M = generate_sparse_matrix(5,5)

add_first_layer (M.A)



