import cv2
import os
import networkx as nx
import matplotlib.pyplot as plt
import utils


name = 'bigvideo.mp4'
print 'opencv version: ' + cv2.__version__

utils.show_video_gray_diff(name)
