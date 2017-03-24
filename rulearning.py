# -*- coding: utf-8 -*
import rugaze
import rugraph as rug
from rugraph_analizer import RuGraphAnalizer

print "--------test-------"
video = 'bigvideo.avi'

gaze = rugaze.VideoSeqGaze(folder_with_videos='dataset', side=3, left_top_coord=(120, 160), log=True)
gaze_auditor = rugaze.GazeHistoryAuditor()
input_layer_shape = gaze.get_shape()
graph = rug.RuGraph(input_layer_shape)

restarts = 0
i=0
while True:
    if i==11935:
        print '-=-=-'
    print str(i) + "----------------------"
    i+=1
    new_frame, was_reseted = gaze.get_next_fixation()
    if was_reseted:
        gaze_auditor.reset()
    if gaze_auditor.do_we_need_shift(new_frame):
        gaze.shift('random')
        continue
    if new_frame is None:
        gaze.restart()
        print "RESTARETD VIDEO FOLDER"
        restarts += 1
        if restarts < 3: # сколько раз проиграть папку с видео
            continue
        else:
            break
    #graph.process_next_input(new_frame, was_reseted)

print "learning ended"
graph.save_droping_accs('test.gexf')

# посмотрим, чему научились нейроны
analizer = RuGraphAnalizer(gaze=gaze, rugraph=graph)
results, counter = analizer.get_nodes_specialisations(graph.get_nodes_of_type('plain'))
analizer.save_results_to_files(results, counter)


