# -*- coding: utf-8 -*
import rugaze
import rugraph as rug

print "--------test-------"
video = 'bigvideo.mp4'

input = rugaze.SimpleVideoGaze(videoname=video, side=10, left_top_coord=(200,200))

input_layer_shape = input.get_shape()
graph = rug.RuGraph(input_layer_shape)


while True:
    new_frame = input.get_next_fixation()
    if new_frame is None:
        break #видео кончилось
    graph.process_next_input(new_frame)

graph.save()
