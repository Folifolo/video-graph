# -*- coding: utf-8 -*
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import utils


# если полезность нейрона ниже пороговой, то удаляем его
DELETE_THR = 0.2

# на сколько за один такт затухает кратковременная память о событии
SALIENCY_FADING_PER_TIME = 0.1

# максимальное количество ярких стабильных нейронов, которое может
# может зарегстрировать в свое рец. поле новодобавлемый NS-нейрон
EVENT_KERNEL_CAPACITY = 6

class GraphError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class RuGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.generator = itertools.count(0)
        self.max_layer = 0

    def _add_input_layer (self, shape):
        for i in range(shape[0]):
            for j in range(shape[1]):
                self._add_input_neuron((i,j))

    def _add_input_neuron(self, index):
        n = self.generator.next()
        self.G.add_node(n,
                        type = "S",
                        layer = 0,
                        activation = 0,
                        index = index
                        )

    def _pass_to_init_layer(self, input_signal):
        for i in range(input_signal.shape[0]):
            for j in range(input_signal.shape[1]):
                n = filter(lambda (n, d): d['index'] == (i,j), self.G.nodes(data=True))
                id = n[0][0] # фильтр всегда находит один нейрон с такими координатами
                self.G.node[id]['activation'] = input_signal[i,j]

    def _add_event_neuron (self, source_ids, source_attrs):
        new_id = self.generator.next()
        self.G.add_node(new_id,
                        type="N",
                        layer= self._get_layer_num_for_neuron(source_attrs),
                        activation=1
                        )
        for id in source_ids:
            self.G.add_edge(id, new_id, weight=source_attrs[id]['activation'])

    def _get_layer_num_for_neuron(self,  source_attrs):
        max_num = 0
        for attrs in source_attrs.values():
            num = attrs['layer']
            if num > max_num:
                max_num = num
        return max_num

    def get_node_weight(self, node_from, node_to):
        return self.G[node_from][node_to]['weight']

    def get_node_activity(self, id):
        return self.G.node[id]['activity']

    def neuron_recognition_rate(self, input_signal, weights):
        return np.cos(input_signal, weights) # число из [0,1]

    def pass_to_neuron(self, id):
        sources = self.G.predecessors(id)
        if len(sources) == 0:
            raise GraphError("Neuron has no input connections")

        input_activities = np.zeros (len(sources))
        input_weights = np.zeros (len(sources))

        for i in range (len(sources)):
            input_weights[i] = self.get_node_weight(sources[i], id)
            input_activities[i] = self.get_node_activity(sources[i])

        self.G.node[id]['activity'] = self.neuron_recognition_rate(input_activities, input_weights)


    def forward_pass(self, input_siganl):
        self._pass_to_init_layer(input_siganl)




    def delete_neuron(self, id):
        self.G.remove_node(id)

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
        for id in ids:
            colors.append(G.node[id]['activation'])

        plt.set_cmap(plt.cm.get_cmap('Blues'))
        nx.draw_networkx_nodes(G, nodelist=ids, pos=positions, node_color=colors)


print "--------test-------"
M = utils.generate_sparse_matrix(15,15)

graph = RuGraph()
graph._add_input_layer( M.A.shape )
graph.forward_pass(M.A)
vis = RuGraphVisualizer()
vis.draw_input_layer_with_act(graph.G)

plt.show()