import matplotlib.pyplot as plt
import networkx as nx

class RuGraphVisualizer:
    def __init__(self):
        pass

    def draw_graph(self, G):
        pass

    def draw_input_layer_with_act(self, G):
        input = nx.get_node_attributes(G, 'index')
        ids = input.keys()
        positions = input.values()
        colors = []
        for n in ids:
            colors.append(G.node[n]['activation'])

        plt.set_cmap(plt.cm.get_cmap('Blues'))
        nx.draw_networkx_nodes(G, nodelist=ids, pos=positions, node_color=colors)