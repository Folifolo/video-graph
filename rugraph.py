# -*- coding: utf-8 -*
import itertools
import math
from collections import deque
import networkx as nx
import ruconsolidator as ruc
import rugraph_inspector
import ru_data_accumulator as acm
from ru_episodic_memory import EpisodicMemory

#константы алгоритма
PREDICTION_THR = 1.0
ACTIVATION_THR = 0.001

# в графе нельзя использовать None, т.к. граф сохраняется в gexf, будет падение.
# поэтому если надо None, то пишем 'None'


class GraphError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

class RuGraph:
    def __init__(self, input_shape, log=True):
        self.iteration = 0
        self.G = nx.DiGraph()
        self.generator = itertools.count(0)
        self.max_layer = -1
        self.log_enabled = log
        self.input_shape = input_shape
        self.episodic_memory = EpisodicMemory(self)
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
        + '\nactive accumulators:' + str(self.episodic_memory.candidates)\
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

    def get_feed_inputs(self, node):
        return [pred for pred in self.G.predecessors(node) if self.G.edge[pred][node]['mtype'] == 'feed']

    def activation_function(self, x):
        return 1 / (1 + math.exp(-x))  # sigmoid

    def get_most_active_nodes(self):
        nodes = [n for n in self.G.nodes() if self.G.node[n]['mtype'] != 'acc' and
                 self.G.node[n]['activation_change'] >= ACTIVATION_THR]
        return nodes

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

    def process_next_input(self, input_signal, was_gaze_reseted):
        self.iteration += 1
        print "--------------------ITERATION " + str(self.iteration) + "--------------------"
        self.print_graph_state()
        self.propagate(input_signal)
        self.prepare_predictions()
        self.episodic_memory.update_accumulators(self.G, was_gaze_reseted)
        self.episodic_memory.consolidation_phase(self.G)
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

