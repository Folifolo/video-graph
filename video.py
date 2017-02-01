import cv2
import os
import networkx as nx
import matplotlib.pyplot as plt

# VIDEO
def print_video_info(videoName):
    if os.path.isfile(name):
        size = os.path.getsize(name)
        print 'video size is ' + str(size) + ' bytes'
    else:
        print 'file doesn\'t exist'
        return

def show_video_grayscale(videoName):
    if not os.path.isfile(name):
        print 'file doesn\'t exist'
        return

    cap = cv2.VideoCapture(name)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret != True:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# GRAPHS

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

### CALL FUNCTIONS
name = 'sample.avi'
print 'opencv version: ' + cv2.__version__

#print_video_info (name)
#show_video_grayscale(name)
sample_graph_networkx()