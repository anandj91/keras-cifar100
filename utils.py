import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from params import *
from models.resnet import resnet50, resnet101, resnet152


def get_mean_std(images):
    mean_channels = []
    std_channels = []

    for i in range(images.shape[-1]):
        mean_channels.append(np.mean(images[:, :, :, i]))
        std_channels.append(np.std(images[:, :, :, i]))

    return mean_channels, std_channels


def pre_processing(train_images, test_images):
    images = np.concatenate((train_images, test_images), axis = 0)
    mean, std = get_mean_std(images)

    for i in range(train_images.shape[-1]):
        train_images[:, :, :, i] = (train_images[:, :, :, i] - mean[i]) / std[i]
        test_images[:, :, :, i] = (test_images[:, :, :, i] - mean[i]) / std[i]
    
    return train_images, test_images


def get_cifar_gen():
    # get dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(
        label_mode='fine')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)

    # preprocess data
    x_train, x_test = pre_processing(x_train, x_test)
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    cifar_gen = datagen.flow(x_train, y_train, batch_size=batch_size)

    testgen = ImageDataGenerator()
    cifar_test_gen = testgen.flow(x_test, y_test, batch_size=batch_size)

    return cifar_gen, cifar_test_gen


def get_model(model_name):
    if model_name == 'resnet50':
        return resnet50()
    elif model_name == 'resnet101':
        return resnet101()
    elif model_name == 'resnet152':
        return resnet152()


def find_lr(epoch_idx, cur_lr):
    if epoch_idx < 60:
        return 0.1
    elif epoch_idx < 120:
        return 0.02
    elif epoch_idx < 160:
        return 0.004
    else:
        return 0.0008

