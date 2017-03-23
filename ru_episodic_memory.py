# -*- coding: utf-8 -*
import itertools
import math
from collections import deque
import networkx as nx
import rugraph
import ruconsolidator as ruc
import rugraph_inspector
import ru_data_accumulator as acm

class EpisodicMemory:
    def __init__(self, log_enabled=True):
        self.num_epizodes = 0
        self.candidates = []
        self.log_enabled = log_enabled

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

    def log(self, message):
        if self.log_enabled:
            print message

    def delete_accumulators(self):
        self.log("deleting all accumulators...")
        accumulators = self.get_nodes_of_type('acc')
        for node_id in accumulators:
            self.G.remove_node(node_id)

    def find_accumulator_to_consolidate(self):
        all_accs = [self.G.node[n]['acc_obj'] for n in self.G.nodes() if self.G.node[n]['mtype'] == 'acc']
        good_accs = itertools.ifilter(lambda acc: acc.is_ready_for_consolidation(), all_accs)
        some_good_acc = next(good_accs, None)
        self.log("acc selected for consolidation: " + str(some_good_acc))
        return some_good_acc # подходит любой из них

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