# -*- coding: utf-8 -*-

import os, struct
import random
import numpy as np
from array import array as pyarray


class Loader:

    def __init__(self):
        pass

    @staticmethod
    def load(dataset='training', digits=np.arange(10), k=None):
        # Get the path for the dataset
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

        # Chose what dataset to open
        if dataset == 'training':
            fname_img = os.path.join(path, 'train-images.idx3-ubyte')
            fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
        elif dataset == 'testing':
            fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
            fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
        else:
            raise ValueError("dataset must be 'testing' or 'training'")

        # Open the label file and parse the content to a pyarray
        flbl = open(fname_lbl, 'rb')
        struct.unpack('>II', flbl.read(8))
        lbl = pyarray('b', flbl.read())
        flbl.close()

        # Open the image file and parse the content to a pyarray
        fimg = open(fname_img, 'rb')
        magic_nr, size, rows, cols = struct.unpack('>IIII', fimg.read(16))
        img = pyarray('B', fimg.read())
        fimg.close()

        # Find the index values for the digits we'd like to include in our image and labels arrays
        if k is None:
            ind = [x for x in range(size) if lbl[x] in digits]
        else:
            # Split the indexes by value
            value_split = [[]] * 10
            for x in range(size):
                value_split[lbl[x] - 1].append(x)

            # Gather the different k values
            ind = []
            for x in range(len(value_split)):
                # Get the random k indexes
                k_random = random.sample(range(0, len(value_split[x])), k)

                # Loop the indexes and get the actual values
                for y in k_random:
                    ind.append(value_split[x][y])

            # Last, but not least, shuffle the final list
            random.shuffle(ind)

        # Get the number of images we're going to parse
        n = len(ind)

        # Create empty images and labels matrices
        images = np.zeros((n, (rows * cols)), dtype=np.uint8)
        labels = np.zeros(n, dtype=np.int8)

        # Loop the index numbers and build the arrays
        for i in range(n):
            images[i] = np.array([0 if x <= 128 else 1 for x in img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]])
            labels[i] = lbl[ind[i]]

        # Return the tuple consisting of images and labels
        return images, labels