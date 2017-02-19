import rugraph as rug
import ruvisualiser as ruv
import utils

print "--------test-------"
graph = rug.RuGraph()
visualiser = ruv.RuGraphVisualizer()

graph._add_input_layer((5, 5))
for i in range(20):
    print "forward pass:"
    M = utils.generate_sparse_matrix(5, 5)
    graph.forward_pass(M.A)

graph.print_info()
visualiser.draw_graph(graph)
