import rugraph as rug
import ruvisualiser as ruv
import utils

print "--------test-------"
graph = rug.RuGraph((10,10))
visualiser = ruv.RuGraphVisualizer()
visualiser.draw_graph(graph)
