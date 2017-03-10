# -*- coding: utf-8 -*
import rugaze
import rugraph as rug

print "--------test-------"
video = 'bigvideo.avi'

input = rugaze.SimpleVideoGaze(video_name=video, side=5, left_top_coord=(210,200))

input_layer_shape = input.get_shape()
graph = rug.RuGraph(input_layer_shape)


while True:
    new_frame = input.get_next_fixation()
    if new_frame is None:
        break # видео кончилось
    graph.process_next_input(new_frame)

graph.save()
