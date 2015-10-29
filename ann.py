# -*- coding: utf-8 -*-

from mnist_loader import Loader

import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet as Tann

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

    def load(self, dataset='training', digits=np.arange(10)):
        """
        Really just encapsulates the load function from the MNIST Loader
        :param dataset: String, the type of dataset to load. Must be either training or testing
        :param digits: Array of list of digits to load. Range 0 - 10
        :return: None
        """

        self.images, self.labels = Loader.load(dataset, digits)

    def build_network(self, layers=[784, 784, 10]):
        # Define the input variable
        input = T.wvector('input')

        # Define the expected result
        expected = T.wvector('expected')

        # Variables holding the weights, bias nd sigmoids
        weights = []
        bias = []
        sigmoids = []

        # Define the different shared variables and the sigmoids
        w1 = theano.shared(np.random.uniform(-.1, .1, size=(784, 784)))
        w1.name = 'Weight 1'

        w2 = theano.shared(np.random.uniform(-.1, .1, size=(784,10)))
        w2.name = 'Weight 2'

        # Define the biases
        b1 = theano.shared(np.random.uniform(-.1, .1, size=784)) # Input width
        b1.name = 'Bias 1'

        b2 = theano.shared(np.random.uniform(-.1, .1, size=10)) # Output width
        b2.name = 'Bias 2'

        # Define the sigmoid for (input * weight1) + bias1
        x1 = Tann.sigmoid(T.dot(input, w1) + b1)
        x1.name = 'Sigmoid 1'

        # Define the sigmoid for (x1 * weight2) + bias2
        x2 = Tann.sigmoid(T.dot(x1, w2) + b2)
        x2.name = 'Sigmoid 2'

        # Define the error calculation
        error = T.sum((expected - x2) ** 2)
        error.name = 'Error'

        # List the params in a array
        params = [w1, b1, w2, b2]

        # Define the gradient
        gradients = T.grad(error, params)

        # Define the backprop function
        backprop_acts = [(p, p - self.learning_rate * g) for p,g in zip(params, gradients)]

        # The predicter, used to run the tests
        self.predictor = theano.function([input], [x2])

        # The trainer, used to train the network
        self.trainer = theano.function([input, expected], [error, x2], updates=backprop_acts)

        # theano.printing.pydotprint(self.trainer,outfile='debug/ann',format='pdf')
        # theano.printing.debugprint(self.trainer)

    def do_training(self, epochs=4):
        num = len(self.images)

        # Debug
        print ''
        print 'Training'
        print 'Size of training set: ' + str(num)
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
        print 'Final error value: ' + str(error)

    def do_testing(self):
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

            #print test_result
            #print guessed_number
            #print self.labels[i][0]
            #print '----'

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



# New instance of the network
an = ANN()

# Load the file
an.load()

# Build the network
an.build_network([784, 784, 9])

# Train once
an.do_training()

# Run the tests!
#an.do_testing()

# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32