# -*- coding: utf-8 -*-

from mnist_loader import Loader

import os
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as Tann
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

srng = RandomStreams()

# Debug on/off
debug = False
if __name__ == '__main__':
    debug = True


class ANN:

    def __init__(self):
        """
        Constructor for ANN class

        :return: None
        """

        # Keep track of the data sets etc
        self.train_images = None
        self.train_labels = None

        self.test_images = None
        self.test_labels = None

        self.test_blind = None

        # The learning rate
        self.learn_rate = 0.0001

        # The train and predict functions
        self.train = None
        self.predict = None

    def load(self, dataset='training'):
        """
        Uses the Loader class to load the images from the binary MNIST files

        :param dataset: Decides the dataset to be loaded. Has to be either training or testing
        :return: None
        """

        if debug:
            print('Loader')
            print('=======================')
            print('Loading dataset of type: ' + dataset)

        # Fetch the set
        images, labels = Loader.load(dataset)

        # Store correctly
        if dataset == 'training':
            self.train_images = images
            self.train_labels = labels
        else:
            self.test_images = images
            self.test_labels = labels

        if debug:
            print('Loaded dataset of size: ' + str(len(self.train_labels)))
            print('')

    def rmsprop(self, error, params, rho=0.9):
        """
        RMS prop, used for updating the weights

        :param error:
        :param params:
        :return:
        """

        grads = T.grad(cost=error, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)

            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + 0.000001)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - self.learn_rate * g))
        return updates

    def build_network(self, layers=[784, 784, 10]):
        """
        Method building the network

        :param layers: List of layer width
        :return: None
        """

        if debug:
            print('Creating Neural Network')
            print('=======================')
            print('Input layer neurons: ' + str(layers[0]))
            for i in range(1, len(layers) - 1):
                print('Hidden layer #' + str(i) + ' neurons: ' + str(layers[i]))
            print('Output layer neurons: ' + str(layers[-1]))
            print('')

        # Define the input variable
        ipt = T.fmatrix('input')

        # Variables holding the weights, activations and noises
        weights = []
        activations = []
        noises = []

        # Build the layers
        for i in range(1, len(layers)):
            weight = theano.shared(np.asarray(np.random.randn(*(layers[i - 1], layers[i])) * 0.01))
            weight.name = 'Weight ' + str(i)
            weights.append(weight)

            if i == 1:
                # Input -> Hidden layer. Rectify
                activation = Tann.relu(T.dot(ipt, weights[-1]))

                # Input -> Hidden layer. Rectify(dropout)
                noise = Tann.relu(T.dot(ipt, weights[-1]))
            elif i == (len(layers) - 1):
                # Hidden layer -> Output. Softmax
                activation = Tann.softmax(T.dot(activations[-1], weights[-1]))

                # Hidden layer -> Output. Softmax(Dropout)
                noise = Tann.softmax(T.dot(noises[-1], weights[-1]))
            else:
                # Hidden layer -> Hidden layer. Rectify
                activation = Tann.relu(T.dot(activations[-1], weights[-1]))

                # Hidden layer -> Hidden layer. Rectify(dropout)
                noise = Tann.relu(T.dot(noises[-1], weights[-1]))

            activation.name = 'Activation ' + str(i)
            activations.append(activation)

            noise.name = 'Noise ' + str(i)
            noises.append(noise)

        # Build params list
        params = []
        for i in range(len(weights)):
            params.append(weights[i])

        # Define the train function
        expected = T.fmatrix('expected')
        error = T.mean(Tann.categorical_crossentropy(noises[-1], expected))
        updates = self.rmsprop(error, params)
        self.train = theano.function(inputs=[ipt, expected], outputs=error, updates=updates, allow_input_downcast=True)

        # Define the predict function
        output = T.argmax(activations[-1], axis=1)
        self.predict = theano.function(inputs=[ipt], outputs=output, allow_input_downcast=True)

    def do_training(self, epochs=1, n=128, run_tests=False, test_interval=1):
        """
        Do the actual training

        :param epochs: Number of epochs to train. Integer
        :param n: Number of datasets to train at the same time. Integer
        :param run_tests: Should run test after each epoch. Boolean
        :param test_interval:  If run_tests is True, define how often we should run the test
        :return: None
        """

        if debug:
            print('Training')
            print('=======================')

        # Make sure we got what we need
        if self.train_images is None:
            print('No training set loaded, aborting')
            return

        # Check if we can run tests in between each epoch
        if run_tests and self.test_images is None:
            print('No testing set loaded, not testing between training epochs')
            print('')

            # Reset run test value
            run_tests = False

        # Debug information
        if debug:
            print('Training total of: ' + str(epochs) + ' epochs')
            print('Size of n: ' + str(n))
            print('Testing between runs: ' + ('Yes' if run_tests else 'No'))
            print('')

        # Run the training set
        errors = []
        for i in range(epochs):
            # Debug print
            print('Epoch #' + str(i + 1))

            # Keep track of the current error value
            error = 0

            # Do the actual training, by passing n number of cases at the time
            for start in range(0, len(self.train_labels), n):
                # Calculate end
                end = start + n
                if end > len(self.train_labels):
                    end = len(self.train_labels)

                # Train n number of images
                error += self.train(self.train_images[start:end], self.train_labels[start:end])

            # Print the error rate
            print('- Error rate: ' + str(error))

            # Check if we should test now
            if run_tests and i % test_interval == 0:
                self.do_testing(output='inline')

            # Newline
            print('')

            # Add error value to errors
            errors.append(error)

    def do_testing(self, output='normal'):
        """
        Do the testing of the NN

        :param mode: Defines how we should do the tests
        :param mode: Define how to output information. Options: normal, inline
        :return: None
        """

        if debug and output == 'normal':
            print('Testing')
            print('=======================')

        # Make sure we have the testing set
        if self.test_images is None:
            print('No testing set loaded, aborting')
            return

        # Debug information
        if debug and output == 'normal':
            print('Testing set size: ' + str(len(self.test_labels)))
            print('')

        # Do the actual testing
        guessed = self.predict(self.test_images)
        solutions = np.argmax(self.test_labels, axis=1)

        # Get the total amount of correct guesses
        correct = 0
        for i in range(len(guessed)):
            if guessed[i] == solutions[i]:
                correct += 1

        # Return the number of corrects
        if output == 'inline':
            print('- Correct guesses: ' + str(round(((correct / float(len(guessed))) * 100), 2)) + '%')
        else:
            print('Correct: ' + str(correct) + ' (' + str(round(((correct / float(len(guessed))) * 100), 2)) + '%)')
            print('Wrong: ' + str(len(guessed) - correct) + ' (' + \
                  str(round((((len(guessed) - correct) / float(len(guessed))) * 100), 2)) + '%)')

    def blind_test(self, feature_sets):
        """
        Does blind testing and returns the results as a array

        :param feature_sets: Multi-dimentional array of test cases
        :return: The guessed values in a list
        """

        return list(self.predict(feature_sets))

# If the script was called directly, run this debug stuff
if debug:
    # Make some room
    print('')

    # New instance of the network
    an = ANN()

    # Load the file(s)
    an.load(dataset='training')
    an.load(dataset='testing')

    # Build the network
    an.build_network([784, 784, 392, 10])

    # Train once
    an.do_training(epochs=20, n=160, run_tests=False)

    # Run the tests!
    an.do_testing()

    # Make some more room
    print('')
