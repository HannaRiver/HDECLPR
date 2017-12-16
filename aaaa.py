from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import h5py
import numpy as np
from PIL import Image
import argparse
import math
import cv2
import os

def discriminator_model():
    model = Sequential()

    model.add(
        Conv2D(64, (5, 5),
        padding='same',
        input_shape=(400, 1200, 3))
    )
    model.add(Activation('relu'))
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.75))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_corossentropy', oprimizer='adam', metrics=['accuracy'])

    return model


def combine_images(tuomo_img, cjh_img):
    #生成图片拼接
    # 将图片resize为 200 * 1200, 不足的部分补 0
    tuomo_resize, cjh_resize = resize_img(tuomo_img, 200, 1200), resize_img(cjh_img, 200, 1200)
    new_img = np.zeros((400, 1200))
    new_img[:200, :] = tuomo_resize
    new_img[200:, :] = cjh_resize
    return new_img

def tagging_images(tuomo_filepaths, cjh_filepaths):
    pass

    
    

def resize_img(img, resize_h, resize_w):
    new_img = np.zeros((resize_h, resize_w))
    img_h, img_w = img.shape[0], img.shape[1]
    ratio_h = int(resize_h / img_h)
    ratio_w = int(resize_w / img_w)
    if ratio_h < ratio_w:
        new_img[:, :ratio_h * img_w] = img.resize(resize_h, ratio_h * img_w)
    else:
        new_img[: ratio_w * img_h, :] = img.resize(ratio_w * img_h, resize_w)
    return new_img


def train_model():
    model = discriminator_model()
    print("Training ……")
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))
    pass
