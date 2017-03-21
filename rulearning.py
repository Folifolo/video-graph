# -*- coding: utf-8 -*
import rugaze
import rugraph as rug
from rugraph_analizer import RuGraphAnalizer

print "--------test-------"
video = 'bigvideo.avi'

gaze = rugaze.VideoSeqGaze(folder_with_videos='dataset', side=20, left_top_coord=(20,60))

input_layer_shape = gaze.get_shape()
graph = rug.RuGraph(input_layer_shape)

i = 0
while True:
    new_frame = gaze.get_next_fixation()
    if new_frame is None:
        # видео кончилось
        gaze.restart()
        i +=1
        if i == 3:
            break
        new_frame = gaze.get_next_fixation()
    graph.process_next_input(new_frame)

print "learning ended"
graph.save_droping_accs('test.gexf')

# посмотрим, чему научились нейроны
analizer = RuGraphAnalizer(gaze=gaze, rugraph=graph)
results, counter = analizer.get_nodes_specialisations(graph.get_nodes_of_type('plain'))
analizer.save_results_to_files(results, counter)


