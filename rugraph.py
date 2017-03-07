# -*- coding: utf-8 -*
import itertools
import math
from collections import deque
import numpy as np
import networkx as nx
import ruconsolidator as ruc
import rugraph_inspector

#константы алгоритма
NEIGHBORHOOD_RADIUS = 4
DESIRED_NUMBER_ENTRYES_PER_OUTCOME = 20
DESIRED_NUMBER_OF_GOOD_OUTCOMES = 2
PREDICTION_THR = 0.7
OUTCOME_LINKING_RADIUS = 10 # макс. расстояние от центра аккумулятора внутри котрого можно искать аутком для связывания
MAX_NUMBER_ACCS_PER_OUTCOME = 100
PERCENTAGE_OF_NODES_TO_REGISTER = 0.3

# Аттрибуты узла - обязательные:
#  type = input, plane, acc
#  has_predict_edges - исходят ли из него хоть одно ребро типа predict (чтоб не перебирать каждый раз всех исходящих)

# Аттрибуты узла - опциональные:
#  input - сумма взвешенных входных сигналов                (для plain)
#  activation - результат применения нелинейности к инпуту  (для plain, input)
#  activation_change = activation(t-1) - activation(t)      (для plain, input)
#  waiting_inputs - сколько инпутов еще должны прилать свои сигналы до того, как можно будет применить функцию активации
#  bias                                                      (для plain)
#  acc_obj                                                   (для acc)
#  acc_node_id  - айдишник узла-аккумулятора, копящего данные для этого узла (для plain, input)

# Аттрибуты ребра - обязательные:
#  weight
#  type: contextual, predict, feed
#
# Аттрибуты ребра - опциональные:
#  если это ребро типа predict, то еще current_prediction


class GraphError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class DataAccumulator:
    def __init__(self, node_id):
        self.ids = []
        self.outcomes_entries = {}
        self.id = node_id
        self.entry_candidate = None

    def _get_entry_for_node(self, G):
        if len(self.ids) == 0:
            self.ids = nx.single_source_shortest_path_length(G, self.id, cutoff=NEIGHBORHOOD_RADIUS).keys()
        entry = []
        num_of_nodes_in_context = 0
        for i in self.ids:
            if G.node[i]['type'] in ['plane', 'input']:
                entry.append(G.node[i]['activation'])
                num_of_nodes_in_context +=1
        assert len(entry) == num_of_nodes_in_context, 'topology changed since the last usage of accumulator, and accum was not erased'
        return entry

    def add_new_entry_candidate(self, G):
        self.entry_candidate = self._get_entry_for_node(G)

    def add_outcome(self, outcome_id):
        assert self.entry_candidate is not None
        self.outcomes_entries[outcome_id] = self.entry_candidate
        self.entry_candidate = None

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

    def get_ids(self):
        return self.ids

    def delete_last_candidate(self):
        self.entry_candidate = None


class RuGraph:
    def __init__(self, input_shape, log=True):
        self.G = nx.DiGraph()
        self.generator = itertools.count(0)
        self.max_layer = -1
        self.log_enabled = log
        self.iteration = 0
        self.input_shape = input_shape
        self.candidates = []    # захешируем айдишники узлов-активных-аккумуляторов
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
                        activation_change=0,
                        type='input',
                        has_predict_edges=False,
                        acc_node_id=None
                        )

    def add_plain_node(self, bias=0):
        node_id = self.generator.next()
        self.G.add_node(node_id,
                        activation=0,
                        activation_change = 0,
                        type='plane',
                        input=0,
                        waiting_inputs=0,
                        has_predict_edges=False,
                        bias=bias,
                        acc_node_id = None
                        )
        return node_id

    def add_acc_node(self, initial_node):
        acc_node_id = self.generator.next()
        acc = DataAccumulator(initial_node)
        self.G.add_node(acc_node_id,
                        type='acc',
                        has_predict_edges=False,
                        acc_obj=acc
                        )
        self.G.add_edge(initial_node, acc_node_id, type='contextual')
        self.G.node[initial_node]['acc_node_id'] = acc_node_id
        return acc_node_id

    def connect_input_weights_to_node(self, node_id, source_nodes_ids, type_of_weights, weights=None):
        for j in range(len(source_nodes_ids)):
            if weights is None:
                weight = 1
            else:
                weight = weights[j]
            self.G.add_edge(source_nodes_ids[j], node_id, weight=weight, type=type_of_weights )

    def connect_output_weights_to_node(self, node_id, target_nodes_ids, type_of_weights, weights=None):
        for j in (len(target_nodes_ids)):
            if weights is None:
                weight = 1
            else:
                weight = weights[j]
            self.G.add_edges_from(node_id, target_nodes_ids[j], weight=weight, type=type_of_weights)
        if type_of_weights == 'predict':
            self.G.node(node_id)['has_predict_edges'] = True

    def init_sensors(self, input_signal):
        assert input_signal.shape == self.input_shape, "input signal has unexpected shape"
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
        self.iteration += 1
        self.log("new iteration:" + str(self.iteration))
        self.init_sensors(input_signal)
        sources = deque(self.get_nodes_of_type('input'))
        sinks = [n for n in self.get_nodes_of_type('plain')]
        # поля input и waiting_inputs нужны только для прямого распространения,
        # надо их очистить от значений с прошлых выховов этий функции
        for n in sinks:
            self.G.node[n]['input'] = 0
            self.G.node[n]['waiting_inputs'] = self.number_of_feed_inputs(n)
        while len(sources) != 0:
            source = sources.popleft()
            activation = self.G.node[source]['activation']
            for target in self.G.successors_iter(source):  # рассылаем от узла сигнал ко всем адрессатам
                if self.G.edge[source][target]['type'] != 'feed':
                    break
                w = self.G.edge[source][target]['weight']
                self.G.node[target]['input'] += activation*w
                if self.G.node[target]['waiting_inputs'] > 1:
                    self.G.node[target]['waiting_inputs'] -= 1
                else:  # этот узел получил данные ото всех, и может теперь сам становиться источником сигнала
                    sinks.remove(target)  # не оч хорошо из листа удалять из произвольного места...
                    sources.append(target)
                    input_to_activation = self.G.node[target]['input'] + self.G.node['bias']
                    new_activation = self.activation_function(input_to_activation)
                    change = activation - new_activation
                    self.G.node[target]['activation'] = new_activation
                    self.G.node[target]['activation_change'] = change
        assert len(sinks) == 0 and len(sources) == 0, \
            "sources and sinks must become empty at the end of propagation, but they did not"
        self.log("propagation done")

    def get_nodes_of_type(self, node_type):
        return [n for n in self.G.nodes() if self.G.node[n]['type'] == node_type]

    def delete_accumulators(self):
        accumulators = self.get_nodes_of_type('acc')
        for node_id in accumulators:
            self.G.remove_node(node_id)

    def number_of_feed_inputs(self, node):
        return len([pred for pred in self.G.predecessors(node) if self.G.edge[pred][node]['type'] == 'feed'])

    def activation_function(self, x):
        return 1 / (1 + math.exp(-x))  # sigmoid

    def find_accumulator_to_consolidate(self):
        all_accs = [self.G.node[n]['acc_obj'] for n in self.G.nodes() if self.G.node[n]['type'] == 'acc']
        good_accs = itertools.ifilter(lambda acc: acc.is_ready_for_consolidation(), all_accs)
        some_good_acc = next(good_accs, None)
        self.log("acc selected: " + str(some_good_acc))
        return some_good_acc # подходит любой из них

    def update_accumulators(self):
        #находим текущие самые яркие (по полю change)
        all_nodes = sorted([n for n in self.G.nodes() if self.G.node[n]['type'] != 'acc'],
                             key=lambda x: self.G.node[x]['activation_change'])
        number_nodes = PERCENTAGE_OF_NODES_TO_REGISTER * len(all_nodes)
        most_active = all_nodes[: int(number_nodes)]
        self.log("update ccumulators: selected " + str(len(most_active)) + " nodes")
        for node in most_active:
            # если изменение активности этого узла на этом такте
            # не было правильно предсказано от прошлого такта
            # значит узел потенциально подходит добавлению в акк в кач-ве ауткома
            if not self.prediction_was_good(node):
                self.try_add_outcome(node)
        # для каждого яркого узла добавляем окрестность узла в акк.
        self.activate_accs(most_active)

    def get_most_relevant_accs_for_outcome(self, node):
        nearest_nodes = nx.single_source_shortest_path_length(self.G, node, cutoff=OUTCOME_LINKING_RADIUS).keys()
        nearest_accs = (n for n in nearest_nodes if n in self.candidates)
        # TODO (1) возможно, перед тем, как брать первые эн штук,стоитх еще посортировать с учетом этой длины пути
        # TODO (2) возможно, еще стоит их посортировать с учетом уровня абстракции (layer)
        return itertools.islice(nearest_accs, MAX_NUMBER_ACCS_PER_OUTCOME)

    def try_add_outcome(self, node_outcome):
        # находим среди активных  аккумуляторов те, котрые находтся достаточно близко к
        # нашему узлу и записываем его в них как аутком
        for acc_id in self.get_most_relevant_accs_for_outcome(node_outcome):
            self.G.node[acc_id]['acc_obj'].add_outcome(node_outcome)

    def clear_last_activity_in_accs(self):
        for acc_node in self.get_nodes_of_type('acc'):
            self.G.node[acc_node]['acc_obj'].delete_last_candidate()

    def activate_accs(self, initial_node_list):
        #сначала удалить активность от прошлого такта из аккумуляторов и из хеша
        self.clear_last_activity_in_accs()
        del self.candidates[:]
        # и после этого уже инициализировать новый набор ждущих аккумуляторов
        # если аккумулятора на узле нет, то создадам его и подсоединим к контекстной окрестности
        for node in initial_node_list:
            acc_for_node = self.G.node[node]['acc_node_id']
            if acc_for_node is None:
                acc_for_node = self.add_acc_node(node)
                ids = self.G.node[acc_for_node]['acc_obj'].get_ids()
                self.connect_input_weights_to_node(node, ids, 'contextual')
            self.G.node[acc_for_node]['acc_obj'].add_new_entry_candidate(self.G)
        self.candidates = initial_node_list

    def calculate_prediction_for_node(self, node_id):
        prediction_input = 0
        for source_id in self.G.predecessors_iter(node_id):
            if self.G.edge[source_id][node_id]['type'] == 'predict':
                w = self.G.edge[source_id][node_id]['weight']
                prediction = self.G.edge[source_id][node_id]['current_prediction']
                prediction_input += w * prediction
        prediction = self.activation_function(prediction_input)
        return prediction

    def prediction_was_good(self, node_id):
        activity_change = self.G.node[node_id]['activation_change']
        old_activity = self.G.node[node_id]['activation']
        prediction = self.calculate_prediction_for_node(node_id)
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
        for node_id in sources_of_predictions:
            activation = self.G.node[node_id]['activation']
            for target_id in self.G.successors_iter(node_id):
                if self.G.edge[node_id][target_id]['type'] == 'predict':
                    self.G.edge[node_id][target_id]['current_prediction'] = activation

    def consolidate(self, accumulator):
        consolidation = ruc.RuConsolidator(accumulator)
        success = consolidation.consolidate()
        if success:
            W1, W2, b1, b2 = consolidation.get_trained_weights()
            source_nodes = accumulator.get_source_nodes()
            sink_nodes = accumulator.get_sink_nodes()
            self.add_new_nodes(W1, W2, b1, b2, source_nodes, sink_nodes)
            return success
        return False  # консолидация не удалась

    def add_new_nodes(self, W1, W2, b1, b2, source_nodes, sink_nodes):
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
        self.prepare_predictions()
        self.update_accumulators()
        accumulator = self.find_accumulator_to_consolidate()
        if accumulator is not None:
            success = self.consolidate(accumulator)
            if success:
                self.delete_accumulators()
        self.inspect_graph()

    def save(self, filename="rugraph.gexf"):
        nx.write_gexf(self.G, filename)

    def inspect_graph(self):
        inspector = rugraph_inspector.RuGraphInspector()
        result = inspector.inspect(self.G)
        if not result:
            raise GraphError(inspector.err_msg)
        else:
            self.log("inspection passed")


