import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sys


class Classifier():
    """Fully Connected Neural Network that uses stochastic gradient descent to minimize a softmax cross entropy objective function with an Adam optimization routine

    Parameters
    ----------
    n_features: int
        number of features each sample has
    n_classes: int
        number of unique classes present in target set
    eta: float
        learning rate
    n_epochs: int
        number of epochs
    random_state: int
        seed used for random number generator
    build: dict
        dictionary specifiying the architecture of the model. n_layers refers to the number of layers, n_units refers to the number of units in each layer and activation_fn refers to the
        activation function to be used in each respective layer. Note that both the n_units and activation_fn must be passed as an iterator, and must appear in the desired order of construction

    """

    def __init__(self, n_features, n_classes, eta=0.01, n_epochs=100, random_state=1, build={'n_layers': 2, 'n_units': [50, 50], 'activation_fn': [tf.nn.relu, tf.nn.relu]}):

        self.eta = eta
        self.n_epochs = n_epochs
        self.random_state = random_state

        g = tf.Graph()

        # the graph is then built using the models 'build' method

        with g.as_default():

            tf.set_random_seed(self.random_state)

            self.build(n_features, n_classes, build, one_hot=False)

            self.init_op = tf.global_variables_initializer()

            self.sess = tf.Session(graph=g)

    def fc_layer(self, name, input_tensor, n_units, activation_fn=None):
        """Function that returns a fully connected node

        Parameters
        ---------
        name: str
            name to be used for the scope/layer
        input_tensor: tensor-object
            the tensor that is passed into the layer
        n_units: int
            number of units present in layer
        activation_fn: function-object
            activation function ot be used for the layer

        Returns
        -------
        layer: tensor-object
            the function returns the evaluated tensor

        """

        with tf.variable_scope(name):

            input_shape = input_tensor.get_shape().as_list()

            weight_shape = (input_shape[1], n_units)

            weight = tf.get_variable(name='weight', shape=weight_shape)
            biases = tf.get_variable(name='bias', initializer=tf.zeros(shape=weight_shape[1]))

            layer = tf.matmul(input_tensor, weight, name='net_input')
            layer = tf.nn.bias_add(layer, biases)

            if activation_fn is not None:
                return activation_fn(layer)
            else:
                return layer

    def build(self, n_features, n_classes, build, one_hot):
        """Function used to construct the graph. Note that the model uses a softmax cross entropy function as the objective function, and an Adam optimization routine to minimize the
        the objective function

        Parameters
        ----------
        n_features: int
            number of features
        n_classes: int
            number of unique classes present in target set
        build: dict
            dictionary specifiying the architecture of the model. n_layers refers to the number of layers, n_units refers to the number of units in each layer and activation_fn refers to the
            activation function to be used in each respective layer. Note that both the n_units and activation_fn must be passed as an iterator, and must appear in the desired order of construction
        one_hot: boolean
            if set to True, the target input is one hot encoded

        """

        tf_x = tf.placeholder(tf.float32, shape=(
            None, n_features), name='tf_x')
        tf_y = tf.placeholder(tf.int32, shape=(None), name='tf_y')

        if one_hot:
            tf_y_onehot = tf.one_hot(
                indices=tf_y, depth=n_classes, name='tf_y_onehot')

        keep_proba = tf.placeholder(tf.float32, name='keep_proba')

        layers = {'h0': tf_x}

        for i in range(build['n_layers']):

            name = 'h{}'.format(i + 1)

            layers[name] = self.fc_layer(name=name, input_tensor=layers['h{}'.format(
                i)], n_units=build['n_units'][i], activation_fn=build['activation_fn'][i])

            layers[name] = tf.nn.dropout(layers[name], keep_proba)

        output = self.fc_layer(
            name='output', input_tensor=layers[name], n_units=n_classes, activation_fn=None)

        y_pred = {'probabilities': tf.nn.softmax(output, name='probabilities'), 'labels': tf.cast(
            tf.argmax(output, axis=1), tf.int64, name='labels')}

        correct_predictions = tf.equal(y_pred['labels'], tf.argmax(tf_y, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=output, labels=tf_y), name='cost')


        train_op = tf.train.AdamOptimizer(
            learning_rate=self.eta).minimize(cost, name='train_op')


    def fit(self, train_set, valid_set=None, batch_size=64):
        """Function used to train the model. Note that, for the spefici problem the model was built for, the log loss was used as a metric. This is not neccesary for the model in general
        and hence can be removed if prefered.

        Parameters
        ----------
        train_set: tuple
            tuple containing training set in form (data_train, target_train)
        valid_set: tuple
            tuple containing validation set in form (data_validation, target_validation)
        batch_size: int
            size of batches to be used in stochastic gradient descent algorithm
        """

        self.sess.run(self.init_op)

        self.training_cost, self.validation_cost = [], []
        self.training_accuracy, self.validation_accuracy = [], []

        if valid_set is None:
            valid_acc = None

        for epoch in range(1, self.n_epochs + 1):

            data_batch = self.batch_generator(data=train_set, batch_size=batch_size, shuffle=True)

            bar = ProgressBar(iter_count=train_set[0].shape[0], batch_size=batch_size)

            for batch_x, batch_y in data_batch:

                bar.update()

                feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y, 'keep_proba:0': 0.5}

                train_acc, train_cost, _ = self.sess.run(['accuracy:0', 'cost:0', 'train_op'], feed_dict=feed)

                self.training_cost.append(train_cost)

            self.training_accuracy.append(train_acc)

            if valid_set is not None:

                feed = {'tf_x:0': valid_set[0], 'tf_y:0': valid_set[1], 'keep_proba:0': 1}

                valid_acc, valid_cost = self.sess.run(['accuracy:0', 'cost:0'], feed_dict=feed)

                self.validation_cost.append(valid_cost)
                self.validation_accuracy.append(valid_acc)

            from sklearn.metrics import log_loss

            loss_feed = {'tf_x:0': train_set[0], 'tf_y:0': train_set[1], 'keep_proba:0': 1}

            y_pred = self.sess.run(['probabilities:0'], feed_dict=loss_feed)[0]

            loss = log_loss(y_true=train_set[1], y_pred=y_pred)

            print('Epoch: {} Avg Train Cost: {:.2f} Train Acc: {:.2f} Validation Acc: {:.2f} Error: {:.2f}'.format(epoch, train_cost, train_acc*100, valid_acc*100, loss))

    def batch_generator(self, data, batch_size=64, shuffle=True):
        """Function used to generate data batches

        Parameters
        ----------
        data: tuple
            tuple of form (data, target)
        batch_size: int
            number of data elements in each batch
        shuffle: boolean
            if set to True, the data is shuffled with each new batch
        """

        X, y = data

        if shuffle:
            p = np.random.permutation(y.shape[0])
            X, y = X[p], y[p]

        for i in range(0, X.shape[0], batch_size):

            X_batch = X[i: (i + batch_size), :]
            y_batch = y[i: (i + batch_size)]

            yield X_batch, y_batch


    def plot_train(self):
        """Function used to plot the cost and accuracy (both of the training and validation set)"""

        from matplotlib import style

        style.use('bmh')

        fig, ax = plt.subplots(nrows=1, ncols=2)

        ax[0].plot(self.training_cost, lw=1, c='r', label='Training Cost')
        ax[0].plot(self.validation_cost, lw=1, c='limegreen', label='Validation Cost', ls='--')
        ax[0].legend(loc='upper right')
        ax[0].set_title('Training Cost')

        ax[1].plot(self.training_accuracy, lw=1, c='r', label='Training Accuracy')
        ax[1].plot(self.validation_accuracy, lw=1, c='limegreen', label='Validation Accuracy', ls='--')
        ax[1].legend(loc='upper right')
        ax[1].set_title('Training Accuracy')

        plt.show()


class ProgressBar():
    """Custom progress bar that displays the progress of the training as the model iterates through the batches

    Parameters
    ----------
    data: array-like
        data that is used for training; note that this refers to the training data before the batches are generated
    bar_length: int
        length of progress bar displayed; purely cosmetic
    batch_size: int
        the number of elements in each batch of data
    """

    def __init__(self, iter_count, bar_length=45, batch_size=64):

        self.iter_count = iter_count
        self.bar_length = bar_length
        self.batch_size = batch_size
        self.updates = 0

    def update(self):
        """Function used to update the progress bar; note that this is done in a loop-like construction and the progress bar needs to be updated with each iteration"""

        end = False

        self.updates += 1

        progress = (self.updates*self.batch_size) / (self.iter_count)

        bars = int(np.floor(self.bar_length*progress))

        if self.updates*self.batch_size < self.iter_count:
            str1 = '[{}/{}]'.format(self.updates*self.batch_size, self.iter_count)
            str2 = '='*(bars - 1) + '>' + '-'*(self.bar_length - bars)

        else:
            str1 = '[{}/{}]'.format(self.iter_count, self.iter_count)
            str2 = '='*(self.bar_length - 1) + '>'
            end = True

        output = '\r{}[{}]'.format(str1, str2)

        sys.stdout.write(output)
        sys.stdout.flush()

        if end:
            print('\n')
