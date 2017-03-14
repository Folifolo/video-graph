# -*- coding: utf-8 -*
import rugaze
import rugraph as rug

print "--------test-------"
video = 'bigvideo.avi'
filename = 'mi.gexf'

input = rugaze.SimpleVideoGaze(video_name=video, side=10, left_top_coord=(120,120),print_it=False )

input_layer_shape = input.get_shape()
graph = rug.RuGraph(input_layer_shape)

i = 0
while True:
    i += 1
    new_frame = input.get_next_fixation()
    if new_frame is None:
        break  # видео кончилось
    graph.process_next_input(new_frame)
    if i > 100:
        break

print "video ended"
graph.save_droping_accs('test.gexf')

