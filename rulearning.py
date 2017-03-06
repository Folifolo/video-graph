import rugaze
import rugraph as rug
import ruvisualiser as ruv

print "--------test-------"
video = 'bigvideo.mp4'

input = rugaze.SimpleVideoGaze(videoname=video,print_it=True, side=10)

input_layer_shape = input.get_shape()
graph = rug.RuGraph(input_layer_shape)


while True:
    new_frame = input.get_next_fixation()
    if new_frame is None:
        break #видео кончилось
    graph.process_next_input(new_frame)

graph.save()
