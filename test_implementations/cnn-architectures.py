#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:58:46 2020

@author: adamcatto
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Activation,\
    Dropout, MaxPool2D, BatchNormalization
from tensorflow.keras.regularizers import l2


# LeNet-5 CNN architecture
def build_lenet_five(activation: str ='tanh'):
    """

    :param activation:
    :return:
    """
    nn = Sequential()
    nn.add(Conv2D(filters=6, kernel_size=5, strides=1, activation=activation,
                  input_shape=(28, 28, 1), padding='same'))
    nn.add(AveragePooling2D(pool_size=2, strides=2, padding='valid'))
    nn.add(Conv2D(filters=16, kernel_size=5, strides=1, activation=activation,
                  input_shape=(14, 14, 6), padding='valid'))
    nn.add(AveragePooling2D(pool_size=2, strides=2, padding='valid'))
    nn.add(Conv2D(filters=120, kernel_size=5, strides=1, activation=activation,
                  input_shape=(5, 5, 16), padding='valid'))
    nn.add(Flatten())
    nn.add(Dense(units=84, activation=activation))
    nn.add(Dense(units=10, activation='softmax'))
    return nn


class LeNet(object):
    def __init__(self, activation: str, lr: float, num_epochs: int, optimizer):
        self.network = build_lenet_five(activation)


class AlexNet(object):
    def __init__(self, activation: str, lr: float, num_epochs: int, optimizer):
        nn = Sequential()

        # layer 1
        nn.add(Conv2D(filters=96, kernel_size=11, strides=4, activation=activation,
                      input_shape=(227, 227, 3), padding='valid'))
        nn.add(MaxPool2D(pool_size=3, strides=2))
        nn.add(BatchNormalization())

        # layer 2
        nn.add(Conv2D(filters=256, kernel_size=5, strides=1, padding='same', kernel_regularizer=l2(5e-4),
                      activation=activation))
        nn.add(MaxPool2D(pool_size=3, strides=2, padding='valid'))
        nn.add(BatchNormalization())

        # layer 3
        nn.add(Conv2D(filters=384, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(5e-4),
                      activation=activation))
        nn.add(BatchNormalization())

        # layer 4
        nn.add(Conv2D(filters=384, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(5e-4),
                      activation=activation))
        nn.add(BatchNormalization())

        # layer 5
        nn.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_regularizer=l2(5e-4),
                      activation=activation))
        nn.add(BatchNormalization())
        nn.add(MaxPool2D(pool_size=3, strides=2))

        # feed into fully-connected layers
        nn.add(Flatten())

        # layer 6
        nn.add(Dense(units=4096, activation=activation))
        nn.add(Dropout(rate=0.5))

        # layer 7
        nn.add(Dense(units=4096, activation=activation))
        nn.add(Dropout(rate=0.5))

        # layer 8 – output
        nn.add(Dense(units=1000, activation='softmax'))

        # initialize
        self.network = nn


net = AlexNet(activation='relu', lr=0.01, num_epochs=10000, optimizer='Adam')
print(net.network.summary())

# TODO: ResNet, GoogLeNet, R-CNN, Inception, VGGNet


# general object detection framework class
class ObjectDetector(object):
    def __init__(self, region_proposal_function, feature_extractor_layers, classifier_layers,
                 non_maximum_suppression, evaluation_metrics):
        self.region_proposal_function = region_proposal_function
        self.feature_extractor_layers = feature_extractor_layers
        self.classifier_layers = classifier_layers
        self.non_maximum_suppression = non_maximum_suppression
        self.evaluation_metrics = evaluation_metrics


def non_maximum_suppression(threshold):
    pass


def iou(ground_truth_box, predicted_box):
    pass