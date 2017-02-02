import cv2
import os
import networkx as nx
import matplotlib.pyplot as plt

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
           diff = pixelwise_diff(prevFrame, frame)
           prevFrame = frame

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

def pixelwise_diff(prev, next):
    diff = prev - next
    return diff

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