# -*- coding: utf-8 -*-

import os
import requests

__mnist_path__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/')

# For reading flat cases from a TEXT file (as provided by Valerij). This is needed for the demo!


def load_flat_text_cases(filename, directory=__mnist_path__):
    f = open(directory + filename, 'r')
    lines = [line.split(" ") for line in f.read().split("\n")]
    f.close()

    x_l = list(map(int, lines[0]))
    x_t = [list(map(int, line)) for line in lines[1:]]

    return x_t, x_l

# You must get this procedure to work PRIOR to the demo session.  It will be swapped out with something very
# similar during the demo session.


def minor_demo(ann, ignore=0):

    def score_it(classification, k=4):
        params = {
            'results': str(ignore) + ' ' + str(classification),
            'raw': '1',
            'k': k
        }

        resp = requests.post('http://folk.ntnu.no/valerijf/5/', data=params)
        return resp.text

    def test_it(ann, cases, k=4):
        images, _ = cases
        predictions = ann.blind_test(images)  # Students must write THIS method for their ANN
        return score_it(predictions, k=k)

    demo100 = load_flat_text_cases('demo100_text.txt')
    training_cases = load_flat_text_cases('all_flat_mnist_training_cases_text.txt')
    test_cases = load_flat_text_cases('all_flat_mnist_testing_cases_text.txt')

    print('TEST Results:')
    print('Training set: \n ', test_it(ann, training_cases, 4))
    print('Testing set:\n ', test_it(ann, test_cases, 4))
    print('Demo 100 set: \n ', test_it(ann, demo100, 8))
