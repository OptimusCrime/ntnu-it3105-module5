# -*- coding: utf-8 -*-

import os
import numpy as np

class Loader:

    def __init__(self):
        pass

    @staticmethod
    def load(dataset='training'):
        # Get the path for the dataset
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

        # Chose what dataset to open
        if dataset == 'training':
            set_img = os.path.join(path, 'train-images.idx3-ubyte')
            set_img_bits = 16
            set_img_shape = (60000, 28 * 28)

            set_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
            set_lbl_bits = 8
            set_lbl_shape = (60000)

        elif dataset == 'testing':
            set_img = os.path.join(path, 't10k-images.idx3-ubyte')
            set_img_bits = 16
            set_img_shape = (10000, 28 * 28)

            set_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
            set_lbl_bits = 8
            set_lbl_shape = (10000)
        else:
            raise ValueError("dataset must be 'testing' or 'training'")


        # Load the images
        set_img_loaded = np.fromfile(file=open(set_img), dtype=np.uint8)
        images = set_img_loaded[set_img_bits:].reshape(set_img_shape).astype(float)

        # Load the labels
        set_lbl_loaded = np.fromfile(file=open(set_lbl), dtype=np.uint8)
        labels_temp = set_lbl_loaded[set_lbl_bits:].reshape(set_lbl_shape)

        # Transform labels
        labels = np.zeros((len(labels_temp), 10), dtype=np.int8)
        for i in range(len(labels_temp)):
            labels[i][labels_temp[i]] = 1


        # Return the collection of images and labels
        return images, labels