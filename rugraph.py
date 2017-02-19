# -*- coding: utf-8 -*
import itertools
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import utils

#сколькими новыми узлами кодировать новое мгновеное восприминание
EPISOD_MAX_SIZE = 7

# размер рецептивного поля
RECEPTIVE_FIELD_SIZE = 6

#порог "узнавания" сигнала слоем
THR_RECOGNITION = 0.9

class GraphError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class RuGraph:
    def __init__(self, log=True):
        self.G = nx.DiGraph()
        self.generator = itertools.count(0)
        self.max_layer = -1
        self.log_enabled = log

    def _add_input_layer (self, shape):
        for i in range(shape[0]):
            for j in range(shape[1]):
                self._add_input_neuron((i,j))
        self.max_layer = 0

    def _add_input_neuron(self, index):
        n = self.generator.next()
        self.G.add_node(n,
                        type="S",
                        layer=0,
                        activation=0,
                        index=index,
                        isInput=True
                        )
###########################################################
########## Прямое распространение сигнала в графе##########
    def propagate_to_init_layer(self, input_signal):
        rows = input_signal.shape[0]
        cols = input_signal.shape[1]
        for i in range(rows):
            for j in range(cols):
                input_neurons = {i:self.G.node[i] for i in self.G.nodes() if self.G.node[i]['layer'] == 0}
                c = [(n, attr) for n, attr in input_neurons.items() if attr['index'] == (i,j)]
                neuron_id = c[0][0] # фильтр всегда находит один нейрон с такими координатами
                self.G.node[neuron_id]['activation'] = input_signal[i,j]

    def get_node_weight(self, node_from, node_to):
        return self.G[node_from][node_to]['weight']

    def get_node_activity(self, id):
        return self.G.node[id]['activation']

    def neuron_recognition_rate(self, input_signal, weights):
        return np.cos(input_signal, weights) # число из [0,1]

    def propagate_to_neuron(self, id):
        sources = self.G.predecessors(id)
        if len(sources) == 0:
            raise GraphError("Neuron has no input connections")
        input_activities = np.zeros (len(sources))
        input_weights = np.zeros (len(sources))
        for i in range (len(sources)):
            input_weights[i] = self.get_node_weight(sources[i], id)
            input_activities[i] = self.get_node_activity(sources[i])
        self.G.node[id]['activity'] = self.neuron_recognition_rate(input_activities, input_weights)

    def delete_neuron(self, id):
        self.G.remove_node(id)

    def propagate_to_layer(self, layer_num):
        for n in self.G.nodes():
            if self.G.node[n]['layer'] == layer_num:
                self.propagate_to_neuron(n)

    def print_info(self):
        print "Number of layers: " + str(self.max_layer + 1)
        print "Number of edges: " + str (self.G.number_of_edges())
        for layer_i in range(self.max_layer + 1):
            n = [n for n in self.G.nodes() if self.G.node[n]['layer'] == layer_i]
            print '   layer ' + str(layer_i) + ": " + str(len(n)) + " nodes;"

    def forward_pass(self, input_signal):
        for layer_i in range(0, self.max_layer + 1):
            if layer_i == 0:
                self.propagate_to_init_layer(input_signal)
            else:
                self.propagate_to_layer(layer_i)

            signal_recognized = self.signal_recognized_by_layer(layer_i)
            if signal_recognized and layer_i != self.max_layer:
                continue
            else:
                # если не удалось построить хорошую низкоуровневую
                # репрезентацию текущего сигнала, то высокоуровневую строить смысла нет
                if not signal_recognized:
                    if layer_i != 0:
                        self.insert_new_neurons_into_layer(layer_i)
                    break
                if layer_i == self.max_layer:
                    self.insert_new_neurons_into_layer(layer_i + 1)

    ###########################################################
    ########## Создание мгновенных воспоминаний###############

    def get_activations_in_layer(self, layer_num):
        return {n: self.get_node_activity(n) for n in self.G.nodes() if self.G.node[n]['layer'] == layer_num}

    def signal_recognized_by_layer(self, layer_num):
        activity_in_layer = self.get_activations_in_layer(layer_num)
        max_val = max(activity_in_layer.values())
        if max_val > THR_RECOGNITION:
            return True
        else:
            return False

    def get_nodes_in_layer(self, layer_num):
        return {n: self.G.node[n] for n in self.G.nodes() if self.G.node[n]['layer'] == layer_num}

    def get_most_active_nodes(self, layer_num):
        all_nodes = self.get_nodes_in_layer(layer_num)
        if len(all_nodes) == 0:
            raise GraphError("unexpected empty layer")
        #узлы слоя в порядке убывания текущей активности
        #sorted_nodes = sorted(all_nodes.items(), key=lambda x: x[1]['activation'], reverse=True)
        active_nodes = {node:attr for node,attr in all_nodes.items() if attr['activation'] > THR_RECOGNITION}
        return active_nodes

    def _get_layer_num_for_neuron(self,  source_neurons):
        max_num = 0
        for neuron in source_neurons:
            num = neuron[1]['layer']
            if num > max_num:
                max_num = num
        return max_num + 1

    def _add_event_neuron (self, neurons):
        new_id = self.generator.next()
        layer_num = self._get_layer_num_for_neuron(neurons)
        self.G.add_node(new_id,
                        type="N",
                        layer=layer_num,
                        activation=1
                        )
        if layer_num > self.max_layer:
            self.max_layer = layer_num
        for n, attr in neurons:
            self.G.add_edge(n, new_id, weight=attr['activation'])

    def insert_new_neurons_into_layer(self, layer_num):
        if layer_num == 0:
            raise GraphError ('attempt to insert new receptors')
        most_active = self.get_most_active_nodes(layer_num - 1)
        lenght = len(most_active)
        if lenght == 0:
            raise GraphError('attempt to insert new neuron without proper input field')
        rec_field_size = math.ceil(float(lenght) / float(EPISOD_MAX_SIZE))
        fields = list(self.chunks(most_active, int(rec_field_size) ))
        for field in fields:
            self._add_event_neuron(field)

    def chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l.items()[i:i + n]

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


print "--------test-------"
graph = RuGraph()
graph._add_input_layer((5, 5))
for i in range(20):
    print "forward pass:"
    M = utils.generate_sparse_matrix(5, 5)
    graph.forward_pass(M.A)

graph.print_info()

