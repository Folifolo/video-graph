# -*- coding: utf-8 -*
import numpy as np
import os
import scipy.misc
import re


# класс позволяет сохранить ввиде картинки то, на что "натаскался" заданный нейрон(ы) графа
class RuGraphAnalizer:

    def __init__(self, gaze, rugraph):
        self.gaze = gaze
        self.graph = rugraph
        self.gaze.restart()

    def get_node_specialisation(self, node, need_save=True):
        sensors_field = self.graph.get_receptive_field_for_node(node)
        result = np.zeros(self.graph.input_shape)
        counter = 0
        while True:
            new_frame = self.gaze.get_next_fixation()
            if new_frame is None:
                break
            self.graph.propagate(new_frame)
            activity_in_node = self.graph.get_node_activity(node)
            what_node_watches_to = self.apply_mask(new_frame, mask=sensors_field)
            result = result + activity_in_node*what_node_watches_to
            counter += 1
        if need_save:
            filename = 'counter_' + str(counter) + '_node_' + str(node) + '.png'
            scipy.misc.toimage(result, cmin=0.0, cmax=1.0).save(filename)
        return result, counter

    def apply_mask(self, matrix, mask):
        for col in range(matrix.shape[1]):
            for row in range(matrix.shape[0]):
                if (row, col) not in mask:
                    matrix[row, col] = 0
        return matrix

    def get_nodes_specialisations(self, nodes):
        sensors_fields = {}
        counter = 0
        results = {}
        for node in nodes:
            sensors_fields[node] = self.graph.get_receptive_field_for_node(node)
            results[node] = np.zeros(self.graph.input_shape)
            counter = 0
        while True:
            new_frame = self.gaze.get_next_fixation()
            if new_frame is None:
                break
            self.graph.propagate(new_frame)
            activity_in_nodes = self.graph.get_nodes_activities(nodes)
            for node in nodes:
                what_node_watches_to = self.apply_mask(new_frame, mask=sensors_fields[node])
                results[node] += activity_in_nodes[node]*what_node_watches_to
            counter += 1
        return results, counter

    def save_results_to_files(self, results, counter):
        folder_name = self.create_folder('res' + str(counter))
        for node in results:
            filename = 'node_'+ str(node)+'.png'
            path = os.path.join(folder_name, filename)
            print path
            scipy.misc.toimage(results[node], cmin=0.0, cmax=1.0).save(path)

    # создаем папку с учетом версии - если папка "имя" сущ-вует, то "имя(0)" и т.д.
    def create_folder(self, name_str='analizer_results'):
        i = 0
        while True:
            if os.path.exists(name_str):
                old = re.search('\(.[0-9]*\)$', name_str)
                if old is not None:
                    old = old.group(0)
                    new = '(' + str(i) + ')'
                    name_str = name_str.replace(old, new)
                else:
                    name_str = name_str + '(' + str(i) + ')'
                i += 1
            else:
                os.makedirs(name_str)
                break
        return name_str

