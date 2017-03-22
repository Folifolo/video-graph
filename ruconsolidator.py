# -*- coding: utf-8 -*
import numpy as np
import theano
import theano.tensor as T
import lasagne

# документация: https://namenaro.gitbooks.io/struktura-proekta/content/chapter1.html
#константы алгоритма
CUCCESS_THR = 0.5 #при какой точности распознавания считать, что обучение удалось
NUM_HIDDEN_UNITS = 3
BATCH_SIZE = 6
LEARNING_RATE = 0.1
NUM_EPOCHS = 200

class RuConsolidator:
    def __init__(self, x, y, log_enabled=True):
        self.log_enabled = log_enabled
        self.X_train = np.array(x)
        self.Y_train = np.array(y)
        self.print_data()
        self.W_in_hid = None
        self.W_hid_out = None
        self.b_in_hid = None
        self.b_hid_out = None

    def log(self, msg):
        if self.log_enabled:
            print "[RuConsolidator] " + msg

    def print_data(self):
        print "X_train:"
        print np.array_str(self.X_train, precision=2)
        print "Y_train:"
        print np.array_str(self.Y_train, precision=2)

    def print_params(self):
        print "Weights input -> hidden:"
        print np.array_str(self.W_in_hid, precision=2)
        print "Weights hidden -> output:"
        print np.array_str(self.W_hid_out, precision=2)
        print "bias hidden:"
        print np.array_str(self.b_in_hid, precision=2)
        print "bias output:"
        print np.array_str(self.b_hid_out, precision=2)

    def _build_model(self, input_var=None):
        classes_num = self.Y_train.shape[1]
        input_data_len = self.X_train.shape[1]
        print str(input_data_len) + " ======================"
        l_in = lasagne.layers.InputLayer(shape=(None, input_data_len),
                                         input_var=input_var)
        l_hidden = lasagne.layers.DenseLayer(l_in, num_units=NUM_HIDDEN_UNITS,
                                             nonlinearity=lasagne.nonlinearities.sigmoid,
                                             W=lasagne.init.GlorotUniform(),
                                             name="hidden_layer")
        print l_hidden.W
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
        input_var = T.matrix(name='inputs')
        target_var = T.matrix(name='targets')

        # символьная оптимизируемая функция
        network = self._build_model(input_var)
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()

        # какие параметры оптимизируем и как
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=LEARNING_RATE, momentum=0.9)
        train_fn = theano.function(inputs=[input_var, target_var],
                                   outputs=loss,
                                   updates=updates,
                                   allow_input_downcast=True) # float64 ->float32
        self.log("consolidation staring...")
        # наконец -- само обучение
        for epoch in range(NUM_EPOCHS):
            train_error = 0
            num_batches = 0
            for batch in self._iterate_minibatches(self.X_train, self.Y_train, BATCH_SIZE, shuffle=True):
                inputs, targets = batch
                train_error += train_fn(inputs, targets)
                num_batches += 1
                avg_err_over_epoch = train_error / num_batches
                self.log("err: " + str(avg_err_over_epoch))
                if avg_err_over_epoch <= 1 - CUCCESS_THR:
                    # да-да, без тестовой части датасета.
                    # тестироваться сетка будет после встраивания в граф - будет предказывать и сверять с истиной
                    success = True
                    self.W_in_hid = lasagne.layers.get_all_param_values(network)[0]
                    self.W_hid_out = lasagne.layers.get_all_param_values(network)[2]
                    self.b_in_hid = lasagne.layers.get_all_param_values(network)[1]
                    self.b_hid_out = lasagne.layers.get_all_param_values(network)[3]
                    break
        if success:
            self.log("SUCESSFULLY!")
        else:
            self.log("consolidation was not successfull")

        #проверка сети
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test = theano.function(inputs=[input_var], outputs=test_prediction, allow_input_downcast= True)
        raw_x = [ 0., 0.1, 0.2, 0.9]
        print raw_x
        print "Classified as: %s" % test([raw_x])


        return success

    def get_trained_weights(self):
        return self.W_in_hid, self.W_hid_out, self.b_in_hid, self.b_hid_out

class Test:
    def __init__(self):
        self.X = [[0.0, 0, 0.1, 0.7],   #1 (1)
                  [0.0, 0, 0.0, 0.9],   #1 (2)
                  [0.0, 0, 0.3, 0.7],   #1 (3)
                  [0.0, 0, 0.2, 0.8],   #1 (4)
                  [0.0, 0, 0.2, 0.9],   #1 (5)
                  [0.0, 0, 0.1, 0.7],   #1 (6)
                  [0.0, 0, 0.2, 0.8],   #1 (7)
                  [0.0, 0, 0.0, 0.9],   #1 (8)
                  [0.0, 0, 0.2, 0.8],   #1 (9)
                  [0.0, 0.0, 0.3, 0.6], #1 (10)
                  [0.0, 0.0, 0.2, 0.9], #1 (11)
                  [0, 0.1, 0.2, 1.0],   #1 (12)
                  [0.8, 0.0, 0.0, 0.0],  #2       (1)
                  [0.9, 0.2, 0.0, 0.0],  #2       (2)
                  [0.7, 0.1, 0.0, 0.0],  #2       (3)
                  [0.8, 0.2, 0.0, 0.0],  #2       (4)
                  [0.9, 0.3, 0.1, 0.0],  #2       (5)
                  [0.8, 0.0, 0.0, 0.0],  #2       (6)
                  [0.8, 0.2, 0.0, 0.0],  #2       (7)
                  [0.8, 0.0, 0.0, 0.0],  #2       (8)
                  [1.0, 0.3, 0.0, 0.0],  #2       (9)
                  [1.0, 0.0, 0.0, 0.0],  #2       (10)
                  [1.0, 0.1, 0.0, 0.0],  #2       (11)
                  [0.9, 0.2, 0.1, 0.0]   #2       (12)
                  ]
        self.Y = [[1, 0],
                  [1, 0],
                  [1, 0],
                  [1, 0],
                  [1, 0],
                  [1, 0],
                  [1, 0],
                  [1, 0],
                  [1, 0],
                  [1, 0],
                  [1, 0],
                  [1, 0],
                  [0, 1],
                  [0, 1],
                  [0, 1],
                  [0, 1],
                  [0, 1],
                  [0, 1],
                  [0, 1],
                  [0, 1],
                  [0, 1],
                  [0, 1],
                  [0, 1],
                  [0, 1]]

    def train(self):
        consolidator = RuConsolidator(x=self.X, y=self.Y)
        res = consolidator.consolidate()
        if res:
            consolidator.print_params()

mtest = Test()
mtest.train()