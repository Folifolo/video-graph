# -*- coding: utf-8 -*
import rugaze
import rugraph as rug
from rugraph_analizer import RuGraphAnalizer

print "--------test-------"
video = 'bigvideo.avi'
folder = 'C:\Users\/neuro/\Downloads/\ASLAN actions similarity database/\ASLAN_AVI'
folder2 = 'C:\Users\/neuro/\Downloads/\ASLAN actions similarity database/\little'
gaze = rugaze.VideoSeqGaze(folder_with_videos=folder2, side=14, left_top_coord=None, log=False)
gaze_auditor = rugaze.GazeHistoryAuditor()
input_layer_shape = gaze.get_shape()
graph = rug.RuGraph(input_layer_shape, log=False)

restarts = 0

while True:
    new_frame, was_reseted = gaze.get_next_fixation()
    if was_reseted:
        gaze_auditor.reset()
    if gaze_auditor.do_we_need_shift(new_frame):
        gaze.shift('random')
        continue
    if new_frame is None:
        restarts += 1
        if restarts < 1: # сколько раз проиграть папку с видео
            gaze.restart()
            continue
        else:
            break
    graph.process_next_input(new_frame, was_reseted)

# распечатаем, сохраним результаты обучения
print "Learning ended..."
graph.log_enabled = True
graph.print_graph_state()
graph.save_droping_accs('test.gexf')

# посмотрим, чему научились нейроны
print "Analize results..."
analizer = RuGraphAnalizer(gaze=gaze, rugraph=graph)
results, counter = analizer.get_nodes_specialisations(graph.get_nodes_of_type('plain'))
analizer.save_results_to_files(results, counter)


