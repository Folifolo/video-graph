# -*- coding: utf-8 -*
import itertools
import ruconsolidator as ruc
import ru_data_accumulator as acm

TOO_MUCH_ACTIVITY = 50
NEIGHBORHOOD_RADIUS = 3
OUTCOME_LINKING_RADIUS = 5  # макс. расстояние от центра аккумулятора внутри котрого можно искать аутком для связывани
ACTIVATION_THR = 0.3

class EpisodicMemory:
    def __init__(self, graph, log_enabled=True):
        self.graph = graph
        self.amount_epizodes_in_memory = 0
        self.candidates = []
        self.log_enabled = log_enabled

    def log(self, message):
        if self.log_enabled:
            print message

    def consolidation_phase(self, G):
        accumulator = self._find_accumulator_to_consolidate(G)
        if accumulator is not None:
            success = self._consolidate(accumulator)
            if success:
                self._delete_accumulators(G)

    def get_best_predicted_and_unpredicted(self, G):
        # возьмем самые активные ноды
        most_active = {n: G.node[n]['episodes_num'] for n in G.nodes() if G.node[n]['mtype'] != 'acc' and
                 G.node[n]['activation_change'] >= ACTIVATION_THR}
        # и посортируем их по заполненности связанных с ними аккумуляторов (по убыванию):
        most_active = sorted(most_active, key=lambda k: most_active[k], reverse=True)
        # активность части из этих нод была предсказана сетью, а части - не была:
        unpredicted = []
        predicted = []
        for node in most_active:
            if not self.graph.prediction_was_good(node):
                unpredicted.append(node)
            else:
                predicted.append(node)
        # если граф большой, то активных нод может быть слишком много, ограничимся же:
        if len(unpredicted) > TOO_MUCH_ACTIVITY:
            unpredicted[:] = unpredicted[0:TOO_MUCH_ACTIVITY+1]
        if len(predicted) > TOO_MUCH_ACTIVITY:
            predicted[:] = predicted[0:TOO_MUCH_ACTIVITY+1]
        return predicted, unpredicted

    def update_accumulators(self, G, was_gaze_reseted):
        predicted, unpredicted = self.get_best_predicted_and_unpredicted(G)
        self.log("before update accumulators: predicted " + str(len(predicted)) + ", unpredicted " + str(len(unpredicted)))
        # если источник входных данных переключился, то связи между
        # прошлыми контекстами и новыми исходами искать не надо:
        if not was_gaze_reseted:
            self._add_as_outcomes(G, unpredicted)
        if len(predicted) < 10:  # если предсказанной активности оч мало, то возьмем всю #TODO 10
            self._add_as_contexts(G, (predicted+unpredicted)[0:TOO_MUCH_ACTIVITY])
        else:
            self._add_as_contexts(G, predicted)

    def _add_acc_node(self, G, initial_node, context_nodes):
        acc_node_id = self.graph.generator.next()
        acc = acm.DataAccumulator(initial_node, context_nodes)
        G.add_node(acc_node_id,
                        mtype='acc',
                        has_predict_edges=False,
                        acc_obj=acc
                    )
        G.add_edge(initial_node, acc_node_id, mtype='contextual')
        G.node[initial_node]['acc_node_id'] = acc_node_id
        return acc_node_id

    def _consolidate(self, accumulator):
        self.log("try to consolidate acuumulator for node " + str(accumulator.id) + "...")
        consolidation = ruc.RuConsolidator(accumulator.get_training_data()[0], accumulator.get_training_data()[1])
        success = consolidation.consolidate()
        if success:
            W1, W2, b1, b2 = consolidation.get_trained_weights()
            source_nodes = accumulator.get_source_nodes()
            sink_nodes = accumulator.get_sink_nodes()
            self.graph.add_new_nodes(W1, W2, b1, b2, source_nodes, sink_nodes)
            return success
        self.log("...consolidation failed.")
        return False  # консолидация не удалась

    def _delete_accumulators(self, G):
        self.log("deleting all accumulators...")
        accumulators = self.graph.get_nodes_of_type('acc')
        for node_id in accumulators:
            G.remove_node(node_id)

    def _find_accumulator_to_consolidate(self, G):
        all_accs = [G.node[n]['acc_obj'] for n in G.nodes()
                    if G.node[n]['mtype'] == 'acc']
        good_accs = itertools.ifilter(lambda acc: acc.is_ready_for_consolidation(), all_accs)
        some_good_acc = next(good_accs, None)
        self.log("acc selected for consolidation: " + str(some_good_acc))
        return some_good_acc # подходит любой из них

    def _add_as_outcomes(self, G, outcomes):
        for acc_id in self.candidates:
            outcome_id = self._find_nearest_from_list(G, acc_id, outcomes)
            if outcome_id is not None:
                G.node[acc_id]['acc_obj'].add_outcome(outcome_id, G)
                self.amount_epizodes_in_memory += 1

    def _find_nearest_from_list(self, G, acc, outcomes):
        center_of_acc = G.node[acc]['acc_obj'].id
        nearest_nodes = self.graph.get_ego_neighborhood(node=center_of_acc, cutoff=OUTCOME_LINKING_RADIUS)
        self.log("looking among nearest.. :" + str(nearest_nodes))
        for k in list(nearest_nodes):
            if k not in outcomes:
                del nearest_nodes[k]
        if len(nearest_nodes) == 0:
            return None
        return max(nearest_nodes, key=lambda x: x[1])

    def _clear_last_activity_in_accs(self, G):
        for acc_node in self.graph.get_nodes_of_type('acc'):
            G.node[acc_node]['acc_obj'].delete_last_candidate()

    def _add_as_contexts(self, G, initial_node_list):
        # сначала удалить активность от прошлого такта из аккумуляторов и из хеша
        self._clear_last_activity_in_accs(G)
        del self.candidates[:]
        # и после этого уже инициализировать новый набор ждущих аккумуляторов
        # если аккумулятора на узле нет, то создадам его и подсоединим к контекстной окрестности
        for i in xrange(len(initial_node_list) - 1, -1, -1):
            node = initial_node_list[i]
            acc_for_node = G.node[node]['acc_node_id']
            if acc_for_node is 'None':
                context_nodes = self.graph.get_ego_neighborhood(node,
                                                          cutoff=NEIGHBORHOOD_RADIUS).keys()
                if len(context_nodes) <= acm.MIN_ENTRY_LEN:
                    del initial_node_list[i]
                    continue
                acc_for_node = self._add_acc_node(G, node, context_nodes)
                ids = G.node[acc_for_node]['acc_obj'].get_ids()
                self.graph.connect_input_weights_to_node(node, ids, 'contextual')
            G.node[acc_for_node]['acc_obj'].add_new_entry_candidate(G)
        self.candidates = [G.node[n]['acc_node_id'] for n in initial_node_list]

