# -*- coding: utf-8 -*
import networkx as nx
import matplotlib.pyplot as plt

# если специализация нейрона ниже пороговой, то удаляем его
DELETE_THR = 0.2

# на сколько за один такт затухает кратковременная память о событии
SALIENCY_FADING_PER_TIME = 0.1

# максимальное количество ярких стабильных нейронов, которое может
# может зарегстрировать в свое рец. поле новодобавлемый NS-нейрон
EVENT_KERNEL_CAPACITY = 6

class RuGraph:
    def __init__(self):
        self.G = nx.DiGraph()

    # инициализировать начальный слой как матризу S-нейронов, ни с чем не соединых
    def add_input_layer (self, shape):
        pass

    def forvard_pass(self, input_siganl):
        pass

    def add_event_neuron (self, source_neurons):
        pass

    def delete_neuron(self, id):
        pass

class RuGraphVisualizer:
    def __init__(self):
        pass

    def draw_graph(self, G):
        pass


print "--------"

graph = RuGraph()
vis = RuGraphVisualizer()
vis.draw_graph(graph)