# -*- coding: utf-8 -*
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
plt.ion()

class RuGraphVisualizer:
    def __init__(self):
        pass

    def draw_graph(self, graph):
        nx.draw_spring(graph.G)
        plt.show()

    def draw_input_layer_with_act(self, G):
        input = nx.get_node_attributes(G, 'index')
        ids = input.keys()
        positions = input.values()
        colors = []
        for n in ids:
            colors.append(G.node[n]['activation'])

        plt.set_cmap(plt.cm.get_cmap('Blues'))
        nx.draw_networkx_nodes(G, nodelist=ids, pos=positions, node_color=colors)


class UpdatingDiagram:
    def __init__(self):
        pass

    def on_launch(self):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([],[], 'o')
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        #Other stuff
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 500)
        self.ax.grid()

    def update(self, x_point, y_point):
        #Update data (with the new _and_ the old points)
        self.lines.set_xdata(np.append(self.lines.get_xdata(), x_point))
        self.lines.set_ydata(np.append(self.lines.get_ydata(), y_point))
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()