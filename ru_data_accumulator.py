# -*- coding: utf-8 -*
import utils

DESIRED_NUMBER_ENTRYES_PER_OUTCOME = 10
DESIRED_NUMBER_OF_GOOD_OUTCOMES = 2
MIN_ENTRY_LEN = 2

class DataAccumulator:
    def __init__(self, node_id, context_nodes, log=False):
        self.ids = context_nodes
        self.outcomes_entries = {}
        self.id = node_id
        self.entry_candidate = None
        self.log_enabled = log

    def log(self, msg):
        if self.log_enabled:
            print "[DataAcc class] " + msg

    def add_new_entry_candidate(self, G):
        assert len(self.ids) > MIN_ENTRY_LEN, "corrupted accumulator " + str(self.id)
        entry = []
        for i in self.ids:
            assert G.node[i]['mtype'] in ['plain', 'input'], "wrong type of context neuron"
            entry.append(G.node[i]['activation'])
        self.entry_candidate = entry

    def add_outcome(self, outcome_id):
        assert self.entry_candidate is not None, "context was not initialized for that event"
        if outcome_id not in self.outcomes_entries:
            self.outcomes_entries[outcome_id] = []
        self.outcomes_entries[outcome_id].append(self.entry_candidate)
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
        X_train = []
        Y_train = []
        for outcome in good_outcomes:
            for entry in self.outcomes_entries[outcome]:
                assert len(entry) > MIN_ENTRY_LEN
                X_train.append(entry)
                Y_train.append(outcome)
        return X_train, Y_train

    def get_ids(self):
        return self.ids

    def delete_last_candidate(self):
        self.entry_candidate = None

    def print_state(self):
        print "AСС: center node " + str(self.id) + "~~~~~~~~~~~~"
        print "entries: " + str(self.outcomes_entries)
        print "active entry: " + str(self.entry_candidate)

