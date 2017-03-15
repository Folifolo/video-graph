import cv2
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def print_video_info(videoName):
    if os.path.isfile(videoName):
        size = os.path.getsize(videoName)
        print 'video size is ' + str(size) + ' bytes'
    else:
        print 'file doesn\'t exist'
        return



def show_video_gray(videoName):
    if not os.path.isfile(videoName):
        print 'file doesn\'t exist'
        return

    cap = cv2.VideoCapture(videoName)
    if not cap.isOpened():
        print "opencv failed to open video, may be ffmpeg is missing?"
    while cap.isOpened():
        ret, frame = cap.read()
        if ret != True:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def show_video_gray_diff(videoName):
    if not os.path.isfile(videoName):
        print 'file doesn\'t exist'
        return

    cap = cv2.VideoCapture(videoName)
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
           diff = pixelwise_diff(prevFrame, frame, 10)
           prevFrame = frame

        draw_anchor(50,50, diff) #TODO moving anchor
        cv2.imshow('diff', diff)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def show_first_n_frames(videoName, n):
    if not os.path.isfile(videoName):
        print 'file doesn\'t exist'
        return

    cap = cv2.VideoCapture(videoName)
    if not cap.isOpened():
        print "opencv failed to open video, may be mmpeg is missing?"
    i=0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret != True:
            break
        cv2.imshow('frame'+str(i), frame)
        if i > n-1 :
            break
        i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

def pixelwise_diff(prev, next, threshold):
    img = np.zeros((prev.shape[0], prev.shape[1], 3), np.uint8) # create black frame 3 channels
    for i in range(prev.shape[0]):
        for j in range(prev.shape[1]):
            pixel_diff = int(prev[i,j]) - int(next[i,j])
            if abs(pixel_diff) > threshold:
                if pixel_diff < 0:
                    set_pixel_blue(i,j,img)
                else:
                    set_pixel_red(i,j, img)

    return img

def draw_anchor(x, y, img):
    color = (255, 255, 255)
    inner_radius = 4
    cv2.circle(img, (int(x), int(y)), inner_radius, color, cv2.FILLED)


def draw_quad(x, y, side, img):
    half = int(side/2)
    pt1 = (x - half, y - half)
    pt2 = (x + half, y + half)
    color = (255,255,255)
    cv2.rectangle(img, pt1, pt2, color)

def set_pixel_red(i,j, img):
    img[i,j,0] = 255

def set_pixel_blue(i,j, img):
    img[i,j,2] = 255

def sample_graph_networkx():
    G = nx.Graph()
    G.add_node('1')
    G.add_node('2')
    G.add_node('3')
    G.add_node('4')
    G.add_node('5')
    G.add_edge('1', '2')
    G.add_edge('2', '3')
    G.add_edge('3', '4')
    G.add_edge('4', '1')
    G.add_edge('4', '5')
    nx.draw_spectral(G)
    plt.show()

def get_k_order_neighborhood(G, node, cutoff):
    waves = {0: [node]}
    for wave_i in range(1, cutoff):
        waves[wave_i] = get_neighbors_for_nodes(G, waves[wave_i-1])
        if wave_i > 2:
            waves[wave_i] = [x for x in waves[wave_i] if x not in waves[wave_i-2] ]
    result = {}
    for wave_i in range(0, cutoff):
        if waves[wave_i] is None or len(waves[wave_i]) == 0:
            break
        for node in waves[wave_i]:
            if node not in result:
                result[node] = wave_i
    return result

def get_neighbors_for_nodes(G, nodes):
    neighbors = []
    for node in nodes:
        for neighbor in nx.all_neighbors(G, node):
            neighbors.append(neighbor)
    return neighbors

import scipy
def generate_sparse_matrix(n, m):
    M = scipy.sparse.random(n, m, density = 0.25 )
    return M

#show_video_gray_diff('bigvideo.mp4')
#show_video_gray('bigvideo.mp4')