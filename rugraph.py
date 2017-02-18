# -*- coding: utf-8 -*
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import utils


# если полезность нейрона ниже пороговой, то удаляем его
THR_DELETE = 0.2

# на сколько за один такт затухает кратковременная память о событии
SALIENCY_FADING_PER_TIME = 0.1

#сколькими новыми узлами кодировать новое мгновеное восприминание
EPISOD_MAX_SIZE = 7

# размер рецептивного поля
RECEPTIVE_FIELD_SIZE = 6

#порог "узнавания" сигнала слоем
THR_RECOGNITION = 0.4

#желаемая разреженность
CODE_DENSITY = 20

class GraphError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class RuGraph:
    def __init__(self, log = True):
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
                        type = "S",
                        layer = 0,
                        activation = 0,
                        index = index
                        )
###########################################################
########## Прямое распространение сигнала в графе##########
    def propagate_to_init_layer(self, input_signal):
        rows = input_signal.shape[0]
        cols = input_signal.shape[1]
        for i in range(rows):
            for j in range(cols):
                n = filter(lambda (n, d): d['index'] == (i,j), self.G.nodes(data=True))
                neuron_id = n[0][0] # фильтр всегда находит один нейрон с такими координатами
                self.G.node[neuron_id]['activation'] = input_signal[i,j]

    def get_node_weight(self, node_from, node_to):
        return self.G[node_from][node_to]['weight']

    def get_node_activity(self, id):
        return self.G.node[id]['activity']

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
        for (n, attr) in self.G.nodes():
            if attr['layer'] == layer_num:
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
                if not signal_recognized:
                # если не удалось построить хорошую низкоуровневую
                # репрезентацию текущего сигнала, то высокоуровневую строить смысла нет
                    self.insert_new_neurons_into_layer(layer_i)
                    break
                if layer_i == self.max_layer:
                    self.insert_new_neurons_into_layer(layer_i + 1)

    ###########################################################
    ########## Создание мгновенных воспоминаний###############

    def get_activations_in_layer(self, layer_num):
        return {n : self.get_node_activity(n) for n in self.G.nodes() if self.G.node['layer'] == layer_num}

    def signal_recognized_by_layer(self, layer_num):
        if layer_num == 0:
            return True
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
        #active_nodes = (node, attr for node,attr in all_nodes if attr['activation'] > THR_RECOGNITION)
        active_nodes = ( node  for node in all_nodes if node.value()['activation'] > THR_RECOGNITION)
        return active_nodes

    def _get_layer_num_for_neuron(self,  source_attrs):
        max_num = 0
        for attrs in source_attrs.values():
            num = attrs['layer']
            if num > max_num:
                max_num = num
        return max_num

    def _add_event_neuron (self, neurons):
        new_id = self.generator.next()
        layer_num = self._get_layer_num_for_neuron(neurons.values())
        self.G.add_node(new_id,
                        type = "N",
                        layer = layer_num,
                        activation = 1
                        )
        if layer_num > self.max_layer:
            self.max_layer = layer_num
        for n, attr in neurons:
            self.G.add_edge(n, new_id, weight=attr['activation'])

    def insert_new_neurons_into_layer(self, layer_num):
        most_active = self.get_most_active_nodes(layer_num - 1)
        rec_field_size = len(most_active) / EPISOD_MAX_SIZE
        fields = list(self.chunks(most_active, rec_field_size))
        for field in fields:
            self._add_event_neuron(field)

    def chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

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
graph._add_input_layer( (15, 15) )
for i in range(10):
    M = utils.generate_sparse_matrix(15, 15)
    graph.forward_pass(M.A)

graph.print_info()
