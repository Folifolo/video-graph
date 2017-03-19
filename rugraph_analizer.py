# -*- coding: utf-8 -*
import numpy as np


class RuGraphAnalizer:

    def __init__(self, gaze, rugraph):
        self.gaze = gaze
        self.graph = rugraph
        self.gaze.restart()

    def get_node_specialisation(self, node):
        sensors_field = self.graph.get_receptive_field_for_node(node)
        result = np.zeros(sensors_field.shape)
        while True:
            new_frame = self.gaze.get_next_fixation()
            if new_frame is None:
                break
            self.graph.propagate(new_frame)
            activity_in_node = self.graph.get_node_activity(node)
            what_node_watches_to = self.apply_mask(new_frame, mask=sensors_field)
            result = result + activity_in_node*what_node_watches_to
        return result

    def apply_mask(self, matrix, mask):
        for col in range(matrix.shape[1]):
            for row in range(matrix.shape[0]):
                if (row,col) not in mask:
                    matrix[row, col] = 0
        return matrix

    def get_all_nodes_specialisations(self):
        pass
