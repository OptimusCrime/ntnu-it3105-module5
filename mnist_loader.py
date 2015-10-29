# -*- coding: utf-8 -*-

import os, struct
import numpy as np
from array import array as pyarray


class Loader:

    def __init__(self):
        pass

    @staticmethod
    def load(dataset, digits):
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

        # DEBUG
        #digits = [1, 2]

        # Find the indez values for the digits we'd like to include in our image and labels arrays
        ind = [k for k in range(size) if lbl[k] in digits]

        # Get the number of images we're going to parse
        n = len(ind)

        # Create empty images and labels matrices
        images = np.zeros((n, (rows * cols)), dtype=np.uint8)
        labels = np.zeros((n, 1), dtype=np.int8)

        # Loop the index numbers and build the arrays
        for i in range(n):
            images[i] = np.array([0 if x <= 128 else 1 for x in img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]])
            labels[i] = lbl[ind[i]]

        # Return the tuple consisting of images and labels
        return images, labels