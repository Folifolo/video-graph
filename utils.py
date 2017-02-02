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
           diff = pixelwise_diff(prevFrame, frame, 20) #TODO threshold
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
    outer_radius = 10
    cv2.circle(img, (x, y), inner_radius, color, cv2.FILLED)
    cv2.circle(img, (x, y), outer_radius, color, 1)

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