# -*- coding: utf-8 -*
import numpy as np
import networkx as nx

NEIGHBORHOOD_RADIUS = 4
DESIRED_NUMBER_ENTRYES_PER_OUTCOME = 20
DESIRED_NUMBER_OF_GOOD_OUTCOMES = 2

class DataAccumulator:
    def __init__(self, node_id):
        self.ids = []
        self.outcomes_entries = {}
        self.id = node_id
        self.entry_candidate = None

    def _get_entry_for_node(self, G):
        if len(self.ids) == 0:
            self.ids = nx.single_source_shortest_path_length(G, self.id, cutoff=NEIGHBORHOOD_RADIUS).keys()
            for n in self.ids:
                if G.node[n]['mtype'] not in ['plane', 'input']:
                     self.ids.remove(n)
        entry = []
        num_of_nodes_in_context = 0
        for i in self.ids:
            if G.node[i]['mtype'] in ['plane', 'input']:
                entry.append(G.node[i]['activation'])
                num_of_nodes_in_context += 1
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

    def print_state(self):
        print "AСС: center node " + str(self.id) + "~~~~~~~~~~~~"
        print "entries: " + str(self.outcomes_entries)
        print "active entry: " + str (self.entry_candidate)
