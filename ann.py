# -*- coding: utf-8 -*-

from mnist_loader import Loader

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as Tann

# Debug on/off
debug = False
if __name__ == '__main__':
    debug = True


class ANN:

    def __init__(self):
        """
        Class constructor

        :return: None
        """

        # Stuff related to data
        self.images = None
        self.labels = None

        # Stuff related to the NN
        self.learning_rate = .1
        self.predictor = None
        self.trainer = None

        # Debug
        if debug:
            print ''

    def load(self, dataset='training', digits=np.arange(10)):
        """
        Really just encapsulates the load function from the MNIST Loader

        :param dataset: String, the type of dataset to load. Must be either training or testing
        :param digits: Array of list of digits to load. Range 0 - 10
        :return: None
        """

        if debug:
            print 'Loading dataset of type: ' + dataset
            print ''

        self.images, self.labels = Loader.load(dataset, digits)

        if debug:
            print 'Loaded dataset'
            print ''

    def build_network(self, layers=[784, 784, 10]):
        """
        Method that builds the network by specifying the dimentions for each layer

        :param layers: List of layer sizes, integers in whole numbers
        :return: None
        """

        # Debug
        if debug:
            print 'Creating Neural Network'
            print ''
            print 'Input layer neurons: ' + str(layers[0])
            for i in range(1, len(layers) - 1):
                print 'Hidden layer #' + str(i) + ' neurons: ' + str(layers[i])
            print 'Output layer neurons: ' + str(layers[-1])
            print ''

        # Define the input variable
        ipt = T.wvector('input')

        # Define the expected result
        expected = T.wvector('expected')

        # Avoid Pyhon complaining
        error = None

        # Variables holding the weights, bias and sigmoids
        weights = []
        biases = []
        sigmoids = []

        # Define the different shared variables and the sigmoids
        for i in range(1, len(layers)):
            # Create weights
            weight = theano.shared(np.random.uniform(-.1, .1, size=(layers[i - 1], layers[i])))
            weight.name = 'Weight ' + str(i)
            weights.append(weight)

            # Create biases
            bias = theano.shared(np.random.uniform(-.1, .1, size=layers[i]))
            bias.name = 'Bias ' + str(i)
            biases.append(bias)

            # Create sigmoids
            if i == 1:
                sigmoid = Tann.sigmoid(T.dot(ipt, weights[-1]) + biases[-1])
            else:
                sigmoid = Tann.sigmoid(T.dot(sigmoids[-1], weights[-1]) + biases[-1])
            sigmoid.name = 'Sigmoid ' + str(i)
            sigmoids.append(sigmoid)

            # Create the error correction
            if i == (len(layers) - 1):
                error = T.sum((expected - sigmoids[-1]) ** 2)
                error.name = 'Error'

        # Build params list
        params = []
        for i in range(len(weights)):
            params.extend([weights[i], biases[i]])

        # Define the gradient
        gradients = T.grad(error, params)

        # Define the backprop function
        backprop_acts = [(p, p - self.learning_rate * g) for p, g in zip(params, gradients)]

        # The predicter, used to run the tests
        self.predictor = theano.function([ipt], [sigmoids[-1]])

        # The trainer, used to train the network
        self.trainer = theano.function([ipt, expected], [error, sigmoids[-1]], updates=backprop_acts)

    def do_training(self, epochs=4):
        """
        Execute training for the training set

        :param epochs: Integer, number of epochs to do
        :return: None
        """

        num = len(self.images)

        # Information
        print 'Training'
        print 'Size of training set: ' + str(num)
        print 'Epochs running: ' + str(epochs)
        print ''

        # Array to store error values
        errors = []
        error = 0

        # Loop the epochs
        for i in range(epochs):
            # Debug
            print 'Epoch ' + str(i + 1)
            print '====================='

            # Set the current error value to 0
            error = 0

            # Loop all the images
            for j in range(num):
                # Debug
                if j % 1000 == 0:
                    print str(j) + ' / ' + str(num)

                # Create expected array
                expected = [0] * 10
                expected[self.labels[j]] = 1

                # Train
                output, real_output = self.trainer(self.images[j], expected)

                # Add error
                error += output

            # Add the error sum to the errors array to save progress
            errors.append(error)

            # Debug
            print ''

        # Debug
        print 'Errors: ' + str(errors)

    def do_testing(self):
        """
        Execute the test suite

        :return: None
        """

        # First, load the testing dataset
        self.load(dataset='testing')

        # Get number of tests
        num = len(self.images)

        # Debug
        print ''
        print 'Testing'
        print 'Size of testing set: ' + str(num)

        # Stats
        wrong = 0
        correct = 0

        # Loop all the test cases
        for i in range(num):
            # Run the predictator with the test
            test_result = self.predictor(self.images[i])

            # Get the value the predictor guessed
            guessed_number = np.argmax(test_result)

            print test_result
            print guessed_number
            print self.labels[i][0]
            print '----'

            # Check if we were correct or not
            if self.labels[i] == guessed_number:
                correct += 1
            else:
                wrong += 1

        # Debug print
        print ''
        print 'Guessed ' + str(correct + wrong) + ' numbers'
        print 'Correct: ' + str(correct) + ' (' + str(round(((correct / float(correct + wrong)) * 100), 2)) + '%)'
        print 'Wrong: ' + str(wrong) + ' (' + str(round(((wrong / float(correct + wrong)) * 100), 2)) + '%)'

# If the script was called directly, run this debug stuff
if debug:
    # New instance of the network
    an = ANN()

    # Load the file
    an.load()

    # Build the network
    an.build_network([784, 784, 10])

    # Train once
    an.do_training(epochs=10)

    # Run the tests!
    an.do_testing()
