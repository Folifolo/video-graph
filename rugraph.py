# -*- coding: utf-8 -*
import itertools
import math
from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import ruconsolidator as ruc
import rugraph_inspector
import ru_data_accumulator as acm
import ruvisualiser as ruvis


#константы алгоритма
PREDICTION_THR = 1.0
OUTCOME_LINKING_RADIUS = 4 # макс. расстояние от центра аккумулятора внутри котрого можно искать аутком для связывания
ACTIVATION_THR = 0.001
NEIGHBORHOOD_RADIUS = 4
TOO_MUCH_ACTIVITY = 40

# в графе нельзя использовать None, т.к. граф сохраняется в gexf, будет падение.
# поэтому если надо None, то пишем 'None'


class GraphError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class RuGraph:
    def __init__(self, input_shape, log=True):
        self.num_epizodes = 0
        self.iteration = 0
        self.G = nx.DiGraph()
        self.generator = itertools.count(0)
        self.max_layer = -1
        self.log_enabled = log
        self.input_shape = input_shape
        self.candidates = []    # захешируем айдишники узлов-активных-аккумуляторов
        self._create_input_layer()

    def _create_input_layer(self):
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
        self.G.add_edges_from(edges, weight=1, mtype='contextual')

    def _add_input_neuron(self, index):
        self.G.add_node(index,
                        activation=0,
                        activation_change=0,
                        mtype='input',
                        has_predict_edges=False,
                        acc_node_id='None'
                        )

    def add_plain_node(self, bias=0):
        node_id = self.generator.next()
        self.G.add_node(node_id,
                        activation=0,
                        activation_change = 0,
                        mtype='plane',
                        input=0,
                        waiting_inputs=0,
                        has_predict_edges=False,
                        bias=bias,
                        acc_node_id='None'
                        )
        return node_id

    def add_acc_node(self, initial_node, context_nodes):
        acc_node_id = self.generator.next()
        acc = acm.DataAccumulator(initial_node, context_nodes)
        self.G.add_node(acc_node_id,
                        mtype='acc',
                        has_predict_edges=False,
                        acc_obj=acc,
                        num_episodes=0
                        )
        self.G.add_edge(initial_node, acc_node_id, mtype='contextual')
        self.G.node[initial_node]['acc_node_id'] = acc_node_id
        return acc_node_id

    def connect_input_weights_to_node(self, node_id, source_nodes_ids, type_of_weights, weights=None):
        for j in range(len(source_nodes_ids)):
            if weights is None:
                weight = 1
            else:
                weight = weights[j]
            self.G.add_edge(source_nodes_ids[j], node_id, weight=weight, mtype=type_of_weights )

    def connect_output_weights_to_node(self, node_id, target_nodes_ids, type_of_weights, weights=None):
        for j in (len(target_nodes_ids)):
            if weights is None:
                weight = 1
            else:
                weight = weights[j]
            self.G.add_edges_from(node_id, target_nodes_ids[j], weight=weight, mtype=type_of_weights)
        if type_of_weights == 'predict':
            self.G.node(node_id)['has_predict_edges'] = True

    def init_sensors(self, input_signal):
        assert input_signal.shape == self.input_shape, "input signal has unexpected shape"
        rows = input_signal.shape[0]
        cols = input_signal.shape[1]
        for i in range(rows):
            for j in range(cols):
                activation_last_tact = float(self.G.node[(i, j)]['activation'])
                self.G.node[(i, j)]['activation'] = float(input_signal[i,j])
                self.G.node[(i, j)]['activation_change'] = float(input_signal[i,j] - activation_last_tact)

    def print_graph_state(self):
        msg = "State: " \
        + "\nMax level: " + str(self.max_layer + 1)\
        + "\nNumber of edges: " + str(self.G.number_of_edges())\
        + "\nacc nodes: " + str(len(self.get_nodes_of_type('acc')))\
        + '\nplain nodes: ' + str(len(self.get_nodes_of_type('plain')))\
        + '\ninput nodes: ' + str(len(self.get_nodes_of_type('input')))\
        + '\nactive accumulators:' + str(self.candidates)\
        #for acc_id in [i for i in self.G.nodes() if self.G.node[i]['mtype'] == 'acc']:
            #self.G.node[acc_id]['acc_obj'].print_state()
        self.log(msg)

    def log(self, message):
        if self.log_enabled:
            print message

    def propagate(self, input_signal):
        self.init_sensors(input_signal)
        sources = deque(self.get_nodes_of_type('input'))
        sinks = [n for n in self.get_nodes_of_type('plain')]
        # поля input и waiting_inputs нужны только для прямого распространения,
        # надо их очистить от значений с прошлых выховов этий функции
        for n in sinks:
            self.G.node[n]['input'] = 0
            self.G.node[n]['waiting_inputs'] = len(self.get_feed_inputs(n))
        while len(sources) != 0:
            source = sources.popleft()
            activation = self.G.node[source]['activation']
            for target in self.G.successors_iter(source):  # рассылаем от узла сигнал ко всем адрессатам
                if self.G.edge[source][target]['mtype'] != 'feed':
                    break
                w = self.G.edge[source][target]['weight']
                self.G.node[target]['input'] += float(activation*w)

                if self.G.node[target]['waiting_inputs'] > 1:
                    self.G.node[target]['waiting_inputs'] -= 1
                else:  # этот узел получил данные ото всех, и может теперь сам становиться источником сигнала
                    sinks.remove(target)  # не оч хорошо из листа удалять из произвольного места...
                    sources.append(target)
                    input_to_activation = self.G.node[target]['input'] + self.G.node['bias']
                    new_activation = self.activation_function(input_to_activation)
                    change = activation - new_activation
                    self.G.node[target]['activation'] = float(new_activation)
                    self.G.node[target]['activation_change'] = float(change)
        assert len(sinks) == 0 and len(sources) == 0, \
            "sources and sinks must become empty at the end of propagation, but they did not"
        self.log("propagation done")


    def get_nodes_of_type(self, node_type):
        return [n for n in self.G.nodes() if self.G.node[n]['mtype'] == node_type]

    def delete_accumulators(self):
        self.log("deleting all accumulators...")
        accumulators = self.get_nodes_of_type('acc')
        for node_id in accumulators:
            self.G.remove_node(node_id)

    def get_feed_inputs(self, node):
        return [pred for pred in self.G.predecessors(node) if self.G.edge[pred][node]['mtype'] == 'feed']

    def activation_function(self, x):
        return 1 / (1 + math.exp(-x))  # sigmoid

    def find_accumulator_to_consolidate(self):
        all_accs = [self.G.node[n]['acc_obj'] for n in self.G.nodes() if self.G.node[n]['mtype'] == 'acc']
        good_accs = itertools.ifilter(lambda acc: acc.is_ready_for_consolidation(), all_accs)
        some_good_acc = next(good_accs, None)
        self.log("acc selected for consolidation: " + str(some_good_acc))
        return some_good_acc # подходит любой из них

    def get_most_active_nodes(self):
        nodes = [n for n in self.G.nodes() if self.G.node[n]['mtype'] != 'acc' and
                 self.G.node[n]['activation_change'] >= ACTIVATION_THR]
        return nodes

    def update_accumulators(self):
        unpredicted = []
        predicted = []
        most_active = self.get_most_active_nodes()
        self.log("update accumulators: most active " + str(len(most_active)) + " nodes: " + str(most_active))
        for node in most_active:
            # если изменение активности этого узла на этом такте
            # не было правильно предсказано от прошлого такта
            # значит узел подходит добавлению в акк в кач-ве ауткома
            if not self.prediction_was_good(node):
                unpredicted.append(node)
            else:
                predicted.append(node)
        self.log("unpredicted are " + str(len(unpredicted)) + ", predicted = " + str(len(predicted)))
        if len(unpredicted) > TOO_MUCH_ACTIVITY:
            unpredicted = unpredicted[0:TOO_MUCH_ACTIVITY]
        self.add_as_outcomes(unpredicted)
        if len(predicted) > TOO_MUCH_ACTIVITY:
            predicted = predicted[0:TOO_MUCH_ACTIVITY]
        else:
            if len(predicted) < 10:  # TODO
                self.add_as_contexts(most_active[0:TOO_MUCH_ACTIVITY])
        self.add_as_contexts(predicted)
        self.log("accumulators updated")

    def add_as_outcomes(self, outcomes):
        for acc_id in self.candidates:
            outcome_id = self.find_nearest_from_list(acc_id, outcomes)
            if outcome_id is not None:
                self.G.node[acc_id]['acc_obj'].add_outcome(outcome_id)
                self.G.node[acc_id]['num_episodes'] += 1
                self.num_epizodes += 1

    def find_nearest_from_list(self, acc, outcomes):
        center_of_acc = self.G.node[acc]['acc_obj'].id
        nearest_nodes = self.get_ego_neighborhood(node=center_of_acc, cutoff=OUTCOME_LINKING_RADIUS)
        self.log("looking among nearest.. :" + str(nearest_nodes))
        for k in list(nearest_nodes):
            if k not in outcomes:
                del nearest_nodes[k]
        if len(nearest_nodes) == 0:
            return None
        return max(nearest_nodes, key=lambda x: x[1])

    def clear_last_activity_in_accs(self):
        for acc_node in self.get_nodes_of_type('acc'):
            self.G.node[acc_node]['acc_obj'].delete_last_candidate()

    def add_as_contexts(self, initial_node_list):
        # сначала удалить активность от прошлого такта из аккумуляторов и из хеша
        self.clear_last_activity_in_accs()
        del self.candidates[:]
        # и после этого уже инициализировать новый набор ждущих аккумуляторов
        # если аккумулятора на узле нет, то создадам его и подсоединим к контекстной окрестности
        for i in xrange(len(initial_node_list) - 1, -1, -1):
            node = initial_node_list[i]
            acc_for_node = self.G.node[node]['acc_node_id']
            if acc_for_node is 'None':
                context_nodes = self.get_ego_neighborhood(node,
                                                          cutoff=NEIGHBORHOOD_RADIUS).keys()
                if len(context_nodes) <= acm.MIN_ENTRY_LEN:
                    del initial_node_list[i]
                    continue
                acc_for_node = self.add_acc_node(node, context_nodes)
                ids = self.G.node[acc_for_node]['acc_obj'].get_ids()
                self.connect_input_weights_to_node(node, ids, 'contextual')
            self.G.node[acc_for_node]['acc_obj'].add_new_entry_candidate(self.G)
        self.candidates = [self.G.node[n]['acc_node_id'] for n in initial_node_list]

    def get_ego_neighborhood(self, node, cutoff, node_types=['plain', 'input']):
        assert self.G.node[node]['mtype'] in node_types, 'we consider node itself also to be a part of it\'s context'
        ego = nx.ego_graph(self.G, node, radius=cutoff, center=True, undirected=True)
        for n in ego.nodes():
            if ego.node[n]['mtype'] not in node_types:
                ego.remove_node(n)
        return nx.shortest_path_length(ego, source=node)


    def calculate_prediction_for_node(self, node_id):
        prediction_input = 0
        somebody_tried_to_predict = False
        for source_id in self.G.predecessors_iter(node_id):
            if self.G.edge[source_id][node_id]['mtype'] == 'predict':
                somebody_tried_to_predict = True
                w = self.G.edge[source_id][node_id]['weight']
                prediction = self.G.edge[source_id][node_id]['current_prediction']
                prediction_input += w * prediction
        if not somebody_tried_to_predict:
            return None  # сеть даже не пыталась предсказать активность в этой ноде
        prediction = self.activation_function(prediction_input)
        return prediction

    def prediction_was_good(self, node_id):
        activity_change = self.G.node[node_id]['activation_change']
        old_activity = self.G.node[node_id]['activation']
        prediction = self.calculate_prediction_for_node(node_id)
        if prediction is None:
            return False
        return self.prediction_fit_reality_in_node(prediction, old_activity + activity_change)

    def prediction_fit_reality_in_node(self, prediction, reality):
        diff = reality - prediction
        if math.fabs(diff) < PREDICTION_THR:
            return True
        return False

    def prepare_predictions(self):
        # предполагаем, что в activation у всех узлов сейчас актуальные значения
        # и просто рассылаем предсказания изо всех узлов, их которых исходят predict-ребра
        # возьмем горизонт предсказания пока - 1 такт.
        # Сами предсказание в сети не распрстаяется дальше одного ребра от источника (пока)
        sources_of_predictions = [n for n in self.G.nodes() if self.G.node[n]['has_predict_edges'] is True]
        self.log ('there are ' + str(len(sources_of_predictions)) + ' sources of prediction')
        for node_id in sources_of_predictions:
            activation = self.G.node[node_id]['activation']
            for target_id in self.G.successors_iter(node_id):
                if self.G.edge[node_id][target_id]['mtype'] == 'predict':
                    self.G.edge[node_id][target_id]['current_prediction'] = activation

    def consolidate(self, accumulator):
        self.log("try to consolidate acuumulator for node " + str(accumulator.id) + "...")
        consolidation = ruc.RuConsolidator(accumulator.get_training_data()[0],accumulator.get_training_data()[1])
        success = consolidation.consolidate()
        if success:
            W1, W2, b1, b2 = consolidation.get_trained_weights()
            source_nodes = accumulator.get_source_nodes()
            sink_nodes = accumulator.get_sink_nodes()
            self.add_new_nodes(W1, W2, b1, b2, source_nodes, sink_nodes)
            return success
        self.log("...consolidation failed.")
        return False  # консолидация не удалась

    def add_new_nodes(self, W1, W2, b1, b2, source_nodes, sink_nodes):
        self.log("adding new nodes...")
        assert W1 is not None and W2 is not None and b1 is not None and b2 is not None, 'corrupted consolidation'
        assert W1.shape()[0] == W2.shape()[1], \
            'shapes of matrices input-to-hidden and hidden-to-output are inconsistent'
        assert len(source_nodes) != 0 and len(sink_nodes) != 0, 'attempt to insert a node without connections'
        num_of_hidden_units = W1.shape()[0]
        assert len(b1) == num_of_hidden_units, 'biases vector is inconstent with neurons number'
        for i in range(num_of_hidden_units):
            node_id = self.add_plain_node(bias=b1[i])
            self.connect_input_weights_to_node(node_id,
                                               source_nodes_ids=source_nodes,
                                               weights=W1[:, i],
                                               type_of_weights='feed')
            self.connect_output_weights_to_node(node_id,
                                                target_nodes_ids=sink_nodes,
                                                weights=W2[i, :],
                                                type_of_weights='predict')
        for i in range(len(sink_nodes)):
            self.G.node[sink_nodes[i]]['bias'] = b2[i]

    def get_receptive_field_for_node(self, node):
        upper_nodes = [node]
        lower_nodes = set()
        counter_of_iterations = 0
        while True:
            for n in upper_nodes:
                lower_nodes |= set(self.get_feed_inputs(n))  # union
            # если в lower_nodes только сенсоры, то вернуть эти сенсоры
            # используем тот факт, что feed-веса входят только в plain-ноды
            def is_plain_node(item):
                return self.G.node[item]['mtype'] == 'plain'
            upper_nodes = filter(is_plain_node, lower_nodes)
            if len(upper_nodes) == 0:
                assert len(lower_nodes) > 0
                return lower_nodes
            else:
                lower_nodes.clear()
            counter_of_iterations += 1
            assert counter_of_iterations <= self.max_layer + 1

    def get_node_activity(self, node):
        return self.G.node[node]['activity']

    def get_nodes_activities(self, nodes):
        return {node: self.G.node[node]['activity'] for node in nodes}

    def show_progress(self):
        self.diagram.update(x_point=self.iteration, y_point=self.num_epizodes)

    def process_next_input(self, input_signal):
        self.iteration += 1
        print "--------------------ITERATION " + str(self.iteration) + "--------------------"
        self.print_graph_state()
        self.propagate(input_signal)
        self.prepare_predictions()
        self.update_accumulators()
        accumulator = self.find_accumulator_to_consolidate()
        if accumulator is not None:
            success = self.consolidate(accumulator)
            if success:
                self.delete_accumulators()
        self.inspect_graph()

    def save_droping_accs(self, filename="rugraph.gexf"):
        accs = (n for n in self.G.nodes() if self.G.node[n]['mtype'] == 'acc')
        for node in accs:
            self.G.node[node]['acc_obj'] = 'None'
        nx.write_gexf(self.G, filename)
        self.log("the graph was saved to file (the accs were cutted off):" + filename)

    def inspect_graph(self):
        inspector = rugraph_inspector.RuGraphInspector()
        result = inspector.inspect(self.G)
        if not result:
            raise GraphError(inspector.err_msg)

