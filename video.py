import cv2
import os
import networkx as nx
import matplotlib.pyplot as plt
import utils


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

utils.print_video_info (name)
utils.show_video_grayscale(name)
sample_graph_networkx()