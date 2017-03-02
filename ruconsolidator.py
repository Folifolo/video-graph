# -*- coding: utf-8 -*
import numpy as np
import theano
import theano.tensor as T
import lasagne

#константы алгоритма
CUSCCESS_THRESHOLD = 0.8 #при какой точности распознавания считать, что обучение удалось
NUM_HIDDEN_UNITS = 4
BATCH_SIZE = 10
LEARNING_RATE = 0.01
NUM_EPOCHS = 100

class RuConsolidator:
    def __init__(self, accumulator):
        self.X_train, self.Y_train = accumulator.get_training_data()
        self.W_in_hid = None
        self.W_hid_out = None
        self.b_in_hid = None
        self.b_hid_out = None

    def _build_model(self, input_var=None):
        input_data_len = self.X_train.shape()[1]
        classes_num = self.Y_train.shape()[0]
        l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE, input_data_len),
                                         input_var=input_var)
        l_hidden = lasagne.layers.DenseLayer(l_in, num_units=NUM_HIDDEN_UNITS,
                                            nonlinearity=lasagne.nonlinearities.sigmoid,
                                            W=lasagne.init.GlorotUniform(),
                                            name="hidden_layer")
        l_out = lasagne.layers.DenseLayer(l_hidden, num_units=classes_num,
                                        nonlinearity=lasagne.nonlinearities.softmax,
                                        name="output_layer")
        return l_out

    #скопипастено не глядя из https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
    def _iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def consolidate(self):
        success = False
        # символьные входные/выходные переменные
        input_var = T.tensor2('inputs')
        target_var = T.ivector('targets')

        # символьная оптимизируемая функция
        network = self._build_model(input_var)
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()

        # какие параметры оптимизируем и как
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=LEARNING_RATE, momentum=0.9)
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        # наконец -- само обучение
        for epoch in range(NUM_EPOCHS):
            train_error = 0.
            num_batches = 0
            for batch in self._iterate_minibatches(self.X_train, self.Y_train, BATCH_SIZE, shuffle=True):
                inputs, targets = batch
                train_error += train_fn(inputs, targets)
                num_batches += 1
                avg_err_over_epoch = train_error / num_batches
                if avg_err_over_epoch <= 1 - CUSCCESS_THRESHOLD:
                    # да-да, без тестовой части датасета.
                    # тестироваться сетка будет после встраивания в граф - будет предказывать и сверять с истиной
                    success = True
                    self.W_in_hid = network.layers_['hidden_layer'].W.get_value()
                    self.W_hid_out = network.layers_['output_layer'].W.get_value()
                    self.b_in_hid = network.layers_['hidden_layer'].b.get_value()
                    self.b_hid_out = network.layers_['output_layer'].b.get_value()
                    break
        return success

    def get_trained_weights(self):
        return self.W_in_hid, self.W_hid_out, self.b_in_hid, self.b_hid_out