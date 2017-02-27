# -*- coding: utf-8 -*
import itertools
import math
from collections import deque
import numpy as np
import networkx as nx

#константы алгоритма
NEIGHBORHOOD_RADUIS = 4

# аттрибуты узла
#  input - сумма взвешенных входных сигналов
#  activation - результат применения нелинейности к инпуту
#  waiting_inputs - сколько инпутов еще должны прилать свои сигналы до того, как можно будет применить функцию активации

# аттрибуты ребра
#  weight
#  type: contextual, predict, feed


class GraphError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


class RuGraph:
    def __init__(self, input_shape, log=True):
        self.G = nx.DiGraph()
        self.generator = itertools.count(0)
        self.max_layer = -1
        self.log_enabled = log
        self.input_shape = input_shape
        self._create_input_layer()

    def _create_input_layer (self):
        rows, cols = self.input_shape[0], self.input_shape[1]
        for i in range(rows):
            for j in range(cols):
                self._add_input_neuron((i,j))
        self.max_layer = 0
        edges = []
        # соединим внуутри строк
        for i in range( rows):
            for j in range(cols - 1):
                edges.append(((i, j), (i, j + 1)))
        # соединим внутри столбцов
        for j in range(cols):
            for i in range( rows - 1):
                edges.append(((i, j), (i + 1, j)))
        self.G.add_edges_from(edges, weight=1, type='contextual')

    def _add_input_neuron(self, index):
        self.G.add_node(index,
                        activation=0,
                        isInput=True
                        )

    def init_sensors(self, input_signal):
        assert input_signal.shape() == self.input_shape(), "input signal has unexpected shape"
        rows = input_signal.shape[0]
        cols = input_signal.shape[1]
        for i in range(rows):
            for j in range(cols):
                self.G.node[(i,j)]['activation'] = input_signal[i,j]

    def print_info(self):
        print "Max level: " + str(self.max_layer + 1)
        print "Number of edges: " + str(self.G.number_of_edges())

    def log(self, message):
        if self.log_enabled:
            print "rugraph msg: "+ message

    def propagate(self, input_signal):
        self.init_sensors(input_signal)
        sources = deque(self.get_sensors())
        sinks = [n for n in self.G.nodes() if self.G.node[n] not in sources]
        # поля input и waiting_inputs нужны только для прямого распространения,
        # надо их очистить от значений с прошлых выховов этий функции
        for n in sinks:
            self.G.node[n]['input'] = 0
            self.G.node[n]['waiting_inputs'] = self.number_of_feed_inputs(n)

        while len(sources) != 0:
            source = sources.popleft()
            activation = self.G.node[source]['activation']
            for target in self.G.successors(source):  # рассылаем от узла сигнал ко всем адрессатам
                w = self.G.edge[source][target]['weight']
                self.G.node[target]['input'] += activation*w
                if self.G.node[target]['waiting_inputs'] > 1:
                    self.G.node[target]['waiting_inputs'] -= 1
                else:  # этот узел получил данные ото всех, и может теперь сам становиться источником сигнала
                    sinks.remove[target]
                    sources.append(target)
                    self.G.node[target]['activation'] = self.activation_function(self.G.node[target]['input'])

        assert len(sinks) == 0 and len(sources) == 0, "sources and sinks must become empty at the end of propagation, but they did not"
        self.log("propagation done")

    def get_sensors_ids(self):
        return [n for n in self.G.nodes() if self.G.node[n]['isInput'] is True]

    def number_of_feed_inputs(self, node):
        return len ([pred for pred in self.G.predecessors(node) if self.G.edge[pred][node]['type'] == 'feed'])

    def activation_function(self, x):
        return 1 / (1 + math.exp(-x))  # sigmoid

#################### функционал аккумуляторов данных####################

    def get_node_neightborhood(self, node):
        return nx.single_source_shortest_path_length(self.G, node, cutoff=NEIGHBORHOOD_RADUIS).keys()





