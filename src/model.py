from keras.layers import *
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from src.model_config import *

import numpy as np

right_input = Input(shape=(img_size, img_size, n_channel))
left_input = Input(shape=(img_size, img_size, n_channel))
face_input = Input(shape=(img_size, img_size, n_channel))
mask_input = Input(shape=(mask_size * mask_size,))

conv1_right = Conv2D(conv1_eye_out, kernel_size=conv1_eye_size)(right_input)
conv1_pool_right = AveragePooling2D(pool_size=pool1_eye_size, strides=pool1_eye_stride)(conv1_right)
conv2_right = Conv2D(conv2_eye_out, kernel_size=conv2_eye_size)(conv1_pool_right)
conv2_pool_right = AveragePooling2D(pool_size=pool2_eye_size, strides=pool2_eye_stride)(conv2_right)
conv3_right = Conv2D(conv3_eye_out, kernel_size=conv3_eye_size)(conv2_pool_right)
conv3_pool_right = AveragePooling2D(pool_size=pool3_eye_size, strides=pool3_eye_stride)(conv3_right)
conv4_right = Conv2D(conv4_eye_out, kernel_size=conv4_eye_size)(conv3_pool_right)
conv4_pool_right = AveragePooling2D(pool_size=pool4_eye_size, strides=pool4_eye_stride)(conv4_right)

conv1_left = Conv2D(conv1_eye_out, kernel_size=conv1_eye_size)(left_input)
conv1_pool_left = AveragePooling2D(pool_size=pool1_eye_size, strides=pool1_eye_stride)(conv1_left)
conv2_left = Conv2D(conv2_eye_out, kernel_size=conv2_eye_size)(conv1_pool_left)
conv2_pool_left = AveragePooling2D(pool_size=pool2_eye_size, strides=pool2_eye_stride)(conv2_left)
conv3_left = Conv2D(conv3_eye_out, kernel_size=conv3_eye_size)(conv2_pool_left)
conv3_pool_left = AveragePooling2D(pool_size=pool3_eye_size, strides=pool3_eye_stride)(conv3_left)
conv4_left = Conv2D(conv4_eye_out, kernel_size=conv4_eye_size)(conv3_pool_left)
conv4_pool_left = AveragePooling2D(pool_size=pool4_eye_size, strides=pool4_eye_stride)(conv4_left)

conv1_face = Conv2D(conv1_eye_out, kernel_size=conv1_eye_size)(face_input)
conv1_pool_face = AveragePooling2D(pool_size=pool1_eye_size, strides=pool1_eye_stride)(conv1_face)
conv2_face = Conv2D(conv2_eye_out, kernel_size=conv2_eye_size)(conv1_pool_face)
conv2_pool_face = AveragePooling2D(pool_size=pool2_eye_size, strides=pool2_eye_stride)(conv2_face)
conv3_face = Conv2D(conv3_eye_out, kernel_size=conv3_eye_size)(conv2_pool_face)
conv3_pool_face = AveragePooling2D(pool_size=pool3_eye_size, strides=pool3_eye_stride)(conv3_face)
conv4_face = Conv2D(conv4_eye_out, kernel_size=conv4_eye_size)(conv3_pool_face)
conv4_pool_face = AveragePooling2D(pool_size=pool4_eye_size, strides=pool4_eye_stride)(conv4_face)

right_out = Reshape([-1, int(np.prod(conv4_pool_right.get_shape()[1:]))])(conv4_pool_right)
left_out = Reshape([-1, int(np.prod(conv4_pool_left.get_shape()[1:]))])(conv4_pool_left)

eyes = Concatenate(axis=-1)([right_out, left_out])
eyes = Dense(fc_eye_size, activation="relu")(eyes)
eyes = Flatten()(eyes)

face = Reshape([-1, int(np.prod(conv4_pool_face.get_shape()[1:]))])(conv4_pool_face)
face = Dense(fc_face_size, activation="relu")(face)
face = Dense(fc2_face_size, activation="relu")(face)
face = Flatten()(face)

face_mask = Dense(fc_face_mask_size, activation="relu")(mask_input)
face_mask = Dense(fc2_face_mask_size, activation="relu")(face_mask)

fc = Concatenate(axis=-1)([eyes, face, face_mask])
fc2 = Dense(fc_size, activation="relu")(fc)
out = Dense(fc2_size, activation="sigmoid")(fc2)

model = Model([left_input, right_input, face_input, mask_input], out)