import cv2
import os
import networkx as nx
import matplotlib.pyplot as plt
import utils
import numpy as np

name1 = 'bigvideo.mp4'
name2 = 'sample.avi'
print 'opencv version: ' + cv2.__version__

#utils.show_first_n_frames(name,2)
utils.show_video_gray_diff(name1)