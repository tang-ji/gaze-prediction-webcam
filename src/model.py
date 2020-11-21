from keras.layers import *
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from src.config import *

import numpy as np

right_input = Input(shape=(eye_size[0], eye_size[1], n_channel))
left_input = Input(shape=(eye_size[0], eye_size[1], n_channel))
face_input = Input(shape=(face_size[0], face_size[1], n_channel))
mask_input = Input(shape=(mask_size[0], mask_size[1], n_channel))

conv1_right = Conv2D(conv1_eye_out, kernel_size=conv1_eye_size, strides=conv1_eye_strides)(right_input)
conv1_pool_right = MaxPooling2D(pool_size=pool1_eye_size, strides=pool1_eye_stride)(conv1_right)
conv2_right = Conv2D(conv2_eye_out, kernel_size=conv2_eye_size, strides=conv2_eye_strides)(conv1_pool_right)
conv2_pool_right = MaxPooling2D(pool_size=pool2_eye_size, strides=pool2_eye_stride)(conv2_right)

conv1_left = Conv2D(conv1_eye_out, kernel_size=conv1_eye_size, strides=conv1_eye_strides)(left_input)
conv1_pool_left = MaxPooling2D(pool_size=pool1_eye_size, strides=pool1_eye_stride)(conv1_left)
conv2_left = Conv2D(conv2_eye_out, kernel_size=conv2_eye_size, strides=conv2_eye_strides)(conv1_pool_left)
conv2_pool_left = MaxPooling2D(pool_size=pool2_eye_size, strides=pool2_eye_stride)(conv2_left)

conv1_face = Conv2D(conv1_face_out, kernel_size=conv1_face_size, strides=conv1_face_strides)(face_input)
conv1_pool_face = MaxPooling2D(pool_size=pool1_face_size, strides=pool1_face_stride)(conv1_face)
conv2_face = Conv2D(conv2_face_out, kernel_size=conv2_eye_size, strides=conv2_face_strides)(conv1_pool_face)
conv2_pool_face = MaxPooling2D(pool_size=pool2_face_size, strides=pool2_face_stride)(conv2_face)
conv3_face = Conv2D(conv3_face_out, kernel_size=conv3_eye_size)(conv2_pool_face)
conv3_pool_face = MaxPooling2D(pool_size=pool3_face_size, strides=pool3_face_stride)(conv3_face)

conv1_mask = Conv2D(conv1_mask_out, kernel_size=conv1_mask_size, strides=conv1_mask_strides)(mask_input)
conv1_pool_mask = MaxPooling2D(pool_size=pool1_mask_size, strides=pool1_mask_stride)(conv1_mask)
conv2_mask = Conv2D(conv2_mask_out, kernel_size=conv2_mask_size, strides=conv2_mask_strides)(conv1_pool_mask)
conv2_pool_mask = MaxPooling2D(pool_size=pool2_mask_size, strides=pool2_mask_stride)(conv2_mask)
conv3_mask = Conv2D(conv3_mask_out, kernel_size=conv3_mask_size)(conv2_pool_mask)
conv3_pool_mask = MaxPooling2D(pool_size=pool3_mask_size, strides=pool3_mask_stride)(conv3_mask)

right_out = Reshape([-1, int(np.prod(conv2_pool_right.get_shape()[1:]))])(conv2_pool_right)
left_out = Reshape([-1, int(np.prod(conv2_pool_left.get_shape()[1:]))])(conv2_pool_left)

eyes = Concatenate(axis=-1)([right_out, left_out])
eyes = Dense(fc_eye_size, activation="relu")(eyes)
eyes = Flatten()(eyes)

face = Reshape([-1, int(np.prod(conv3_pool_face.get_shape()[1:]))])(conv3_pool_face)
face = Dense(fc_face_size, activation="relu")(face)
face = Dense(fc2_face_size, activation="relu")(face)
face = Flatten()(face)

mask = Reshape([-1, int(np.prod(conv3_pool_mask.get_shape()[1:]))])(conv3_pool_mask)
mask = Dense(fc_mask_size, activation="relu")(mask)
mask = Flatten()(mask)

fc = Concatenate(axis=-1)([eyes, face, mask])
fc2 = Dense(fc_size, activation="relu")(fc)
out = Dense(fc2_size, activation="sigmoid")(fc2)

model = Model([left_input, right_input, face_input, mask_input], out)