# -*- coding: utf-8 -*
from rugraph import DataAccumulator
import numpy as np
import theano
import theano.tensor as T
import lasagne

#константы алгоритма
CUSCCESS_THRESHOLD = 0.8 #при какой точности распознавания считать, что обучение удалось

class RuConsolidator:
    def __init__(self, accumulator):
        self.shape_of_input
        self.shape_of_hidden
        self.shape_of_output

    def build_model(self):
        nonlinearity = lasagne.nonlinearities.sigmoid

    def consolidate(self):
        success = False
        return success

    def get_trained_weights(self):
        W_input_hidden, W_hidden_output = []
        return W_input_hidden, W_hidden_output