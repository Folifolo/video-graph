# -*- coding: utf-8 -*
import itertools
import math
from collections import deque
import numpy as np
import networkx as nx
import ruconsolidator as ruc

#константы алгоритма
NEIGHBORHOOD_RADUIS = 4
DESIRED_NUMBER_ENTRYES_PER_OUTCOME = 20
DESIRED_NUMBER_OF_GOOD_OUTCOMES = 2

# аттрибуты узла
#  input - сумма взвешенных входных сигналов
#  activation - результат применения нелинейности к инпуту
#  waiting_inputs - сколько инпутов еще должны прилать свои сигналы до того, как можно будет применить функцию активации
#  type = input, accum, plane
#  has_predict_edges - исходят ли из него хоть одно ребро типа predict (чтоб не перебирать каждый раз всех исходящих)
#  bias

# аттрибуты ребра
#  weight
#  type: contextual, predict, feed
#  если это ребро типа predict, то еще current_predict


class GraphError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class DataAccumulator:
    def __init__(self, ids, id):
        self.ids = []
        self.outcomes_entries = {}
        self.id = id

    def add_new_entry(self, node_id, G, outcome_id):
        if len(self.ids) == 0:
            self.ids = nx.single_source_shortest_path_length(G, node_id, cutoff=NEIGHBORHOOD_RADUIS).keys()
        entry = []
        for i in self.ids:
            entry.append(G.node[i]['activation'])
        assert len(entry) == len(self.ids), 'topology changed since the last usage of accumulator, and accum was not erased'
        self.outcomes_entries[outcome_id].append(entry)

    def _get_good_outcomes(self):
        outcomes = self.outcomes_entries.keys()
        good_outcomes = []
        for outcome in outcomes:
            if len(self.outcomes_entries[outcome]) >= DESIRED_NUMBER_ENTRYES_PER_OUTCOME:
                good_outcomes.append(outcome)
        return good_outcomes

    def is_ready_for_consolidation(self):
        good_outcomes = self._get_good_outcomes()
        if len(good_outcomes) >= DESIRED_NUMBER_OF_GOOD_OUTCOMES:
            return True
        return False

    def get_training_data(self):
        good_outcomes = self._get_good_outcomes()
        X_train, Y_train = []
        for outcome in good_outcomes:
            for entry in self.outcomes_entries[outcome]:
                X_train.append(entry)
                Y_train.append(outcome)
        return np.array(X_train), np.array(Y_train)



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
                        type='input',
                        has_predict_edges=False
                        )

    def add_new_node(self, bias=0):
        node_id = self.generator.next()
        self.G.add_node(node_id,
                    activation=0,
                    type='plane',
                    input=0,
                    waiting_inputs=0,
                    has_predict_edges=False,
                    bias=bias
                    )
        return node_id

    def connect_input_weights_to_node(self, node_id, source_nodes_ids, weights, type_of_weights):
        for j in range(len(source_nodes_ids)):
            weight = weights[j]
            self.G.add_edge(source_nodes_ids[j], node_id, weight=weight, type=type_of_weights )

    def connect_output_weights_to_node(self, node_id, target_nodes_ids, weights, type_of_weights):
        for j in (len(target_nodes_ids)):
            weight = weights[j]
            self.G.add_edges_from(node_id, target_nodes_ids[j], weight=weight, type=type_of_weights)
        if type_of_weights == 'predict':
            self.G.node(node_id)['has_predict_edges'] = True

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
        # TODO переделать так, чтоб еще считалось поле activity_change на основе текущей и ново-посчитанной activity
        self.init_sensors(input_signal)
        sources = deque(self.get_nodes_of_type('input'))
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
                    input_to_activation = self.G.node[target]['input']+self.G.node['bias']
                    self.G.node[target]['activation'] = self.activation_function(input_to_activation)
        assert len(sinks) == 0 and len(sources) == 0, \
            "sources and sinks must become empty at the end of propagation, but they did not"
        self.log("propagation done")

    def get_nodes_of_type(self, node_type):
        return [n for n in self.G.nodes() if self.G.node[n]['type'] == node_type]

    def delete_accumulators(self):
        accumulators = self.get_nodes_of_type('accumulator')
        for node_id in accumulators:
            self.G.remove_node(node_id)

    def number_of_feed_inputs(self, node):
        return len([pred for pred in self.G.predecessors(node) if self.G.edge[pred][node]['type'] == 'feed'])

    def activation_function(self, x):
        return 1 / (1 + math.exp(-x))  # sigmoid

    def find_accumulator_to_consolidate(self):
        accs = self.get_nodes_of_type('accumulator')
        good_accs = itertools.ifilter(lambda acc: acc.is_ready_for_consolidation(), accs)
        if len(good_accs) == 0:
            return None
        return good_accs[0] # подходит любой из них

    def update_accumulators(self):
        #TODO
        #находим текущие самые яркие (по полю change)
        most_active = sorted([n for n in self.G.nodes()], some labda...)

        for node in most_active:
            # если изменение активности этого узла на этом такте
            # не было правильно предсказано от прошлого такта
            if not self.prediction_was_good(node):
                # значит узел потенциально подходит добавлению в акк в кач-ве ауткома
                for accumulator in self.get_accs_from_past(): #те, у который аутком==unknown с прошлого такта
                    accumulator.try_add_outcome(node, self.G) # внутри этой ф-ции можно поэкспериментровать с условиями добавления


        # для каждого яркого узла добавляем окрестность узла в акк.
        #  В кач-ве ауткома пишем туда unknown. Если  акка нет, то создаем
        self.add_unknouns_to_accs(most_active)

    def add_unknouns_to_accs(self, node_list):
        #сначаала удалить уже существующие accs_from_past из графа и из хеша
        # и после этого уже инициализировать новый набор ждущих аккумуляторов
        for node in node_list:
            ...
        #TODO

    def get_accs_from_past(self):
        # TODO

    def prediction_was_good(self):
        #учимся предсказывать изменения активити, а не саму активити!
        #TODO

    def update_predictions(self):
        # предполагаем, что в activation у всех узлов сейчас актуальные значения
        # и просто рассылаем предсказания изо всех узлов, их которых исходят predict-ребра
        sources_of_predictions = [n for n in self.G.nodes() if self.G.node[n]['has_predict_edges'] is True]
        for node_id in sources_of_predictions:
            activation = self.G.node[node_id]['activation']
            for target_id in self.G.successors_iter(node_id):
                #TODO
                pass


    def consolidate(self, accumulator):
        consolidation = ruc.RuConsolidator(accumulator)
        success = consolidation.consolidate()
        if success:
            W1, W2, b1, b2 = consolidation.get_trained_weights()
            source_nodes = accumulator.get_source_nodes()
            sink_nodes = accumulator.get_sink_nodes()
            self.add_new_nodes(W1, W2, b1, b2, source_nodes, sink_nodes)
            return success
        return False #консолидация не удалась

    def add_new_nodes(self, W1, W2, b1, b2, source_nodes, sink_nodes):
        assert W1 is not None and W2 is not None and b1 is not None and b2 is not None, 'corrupted consolidation'
        assert W1.shape()[0] == W2.shape()[1], \
            'shapes of matrices input-to-hidden and hidden-to-output are inconsistent'
        assert len(source_nodes) != 0 and len(sink_nodes) != 0, 'attempt to insert a node without connections'
        num_of_hidden_units = W1.shape()[0]
        assert len(b1) == num_of_hidden_units, 'biases vector is inconstent with neurons number'
        for i in range(num_of_hidden_units):
            node_id = self.add_new_node(bias=b1[i])
            self.connect_input_weights_to_node(node_id,
                                               source_nodes_ids=source_nodes,
                                               weights=W1[:,i],
                                               type_of_weights='feed')
            self.connect_output_weights_to_node(node_id,
                                                target_nodes_ids=sink_nodes,
                                                weights=W2[i,:],
                                                type_of_weights='predict')
        for i in range(len(sink_nodes)):
            self.G.node[sink_nodes[i]]['bias'] = b2[i]

    def process_next_input(self, input_signal):
        self.propagate(input_signal)
        self.update_accumulators()
        self.update_predictions()
        accumulator = self.find_accumulator_to_consolidate()
        if accumulator is not None:
            success = self.consolidate(accumulator)
            if success:
                self.delete_accumulators()

    def save(self, filename="rugraph.gexf"):
        nx.write_gexf(self.G, filename)






