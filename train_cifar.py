from __future__ import print_function
from enum import Flag
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.models import load_model

import numpy as np
import os
from model import resnet_v1
from utils import *
import random

random.seed(FLAGS.seed)
np.random.seed(FLAGS.seed)
tf.random.set_seed(FLAGS.seed)

# Training parameters
epochs = 200

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Computed depth from supplied model parameter n
n = 5
depth = n * 6 + 2
version = 1

# Model name, depth and version
model_type = "ResNet%dv%d" % (depth, version)

# Load the data.
if FLAGS.dataset == "cifar10":
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
elif FLAGS.dataset == "cifar100":
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
print("y_train shape:", y_train.shape)

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 150:
        lr *= 1e-2
    elif epoch > 100:
        lr *= 1e-1
    print("Learning rate: ", lr)
    return lr


def create_model():
    model_input = Input(shape=input_shape)

    model_dic = {}
    model_out = []
    for i in range(FLAGS.num_models):
        model_dic[str(i)] = resnet_v1(
            input=model_input,
            depth=depth,
            num_classes=num_classes,
            dataset=FLAGS.dataset,
        )
        model_out.append(model_dic[str(i)][2])

    model_output = tf.keras.layers.concatenate(model_out)

    model = Model(inputs=model_input, outputs=model_output)
    initial_epoch = 0

    return model, initial_epoch


def load_last_model():
    import re

    if FLAGS.model_dir == "":
        raise Exception(
            "Invalid args, need to know model directory to load model from last checkpoint"
        )

    load_dir = os.path.join(
        FLAGS.model_dir,
        "seed_" + str(FLAGS.seed),
    )

    load_dir = os.path.join(
        load_dir,
        FLAGS.dataset
        + "_EE_LED_saved_models"
        + str(FLAGS.num_models)
        + "_lamda"
        + str(FLAGS.lamda)
        + "_logdetlamda"
        + str(FLAGS.log_det_lamda)
        + "_"
        + str(FLAGS.augmentation),
    )

    model_names = os.listdir(load_dir)
    last_epoch = max(
        [int(re.match("model+\.([0-9]+)", name).groups()[0]) for name in model_names]
    )
    last_checkpointed_model_name = f"model.{last_epoch}.h5"
    load_path = os.path.join(
        load_dir,
        last_checkpointed_model_name,
    )

    model = load_model(load_path, compile=False)
    return model, last_epoch


model, initial_epoch = (
    create_model() if not FLAGS.load_from_checkpoint else load_last_model()
)

# Prepare model model saving directory.
save_dir = os.path.join(
    os.getcwd() if FLAGS.model_dir == "" else FLAGS.model_dir,
    "seed_" + str(FLAGS.seed),
)

save_dir = os.path.join(
    save_dir,
    FLAGS.dataset
    + "_EE_LED_saved_models"
    + str(FLAGS.num_models)
    + "_lamda"
    + str(FLAGS.lamda)
    + "_logdetlamda"
    + str(FLAGS.log_det_lamda)
    + "_"
    + str(FLAGS.augmentation),
)

model_name = "model.{epoch:03d}.h5"
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

model.compile(
    loss=Loss_withEE_DPP,
    optimizer=Adam(lr=lr_schedule(0)),
    metrics=[acc_metric, Ensemble_Entropy_metric, log_det_metric],
)
model.summary()
print(model_type)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(
    filepath=filepath,
    monitor="val_acc_metric",
    mode="max",
    verbose=2,
    save_best_only=True,
)

lr_scheduler = LearningRateScheduler(lr_schedule)

callbacks = [checkpoint, lr_scheduler]

# Augment labels
y_train_2 = []
y_test_2 = []
for _ in range(FLAGS.num_models):
    y_train_2.append(y_train)
    y_test_2.append(y_test)
y_train_2 = np.concatenate(y_train_2, axis=-1)
y_test_2 = np.concatenate(y_test_2, axis=-1)


# Run training, with or without data augmentation.
if not FLAGS.augmentation:
    print("Not using data augmentation.")
    model.fit(
        x_train,
        y_train_2,
        batch_size=FLAGS.batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test_2),
        shuffle=True,
        verbose=1,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
    )
else:
    print("Using real-time data augmentation.")
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.0,
        # set range for random zoom
        zoom_range=0.0,
        # set range for random channel shifts
        channel_shift_range=0.0,
        # set mode for filling points outside the input boundaries
        fill_mode="nearest",
        # value used for fill_mode = "constant"
        cval=0.0,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0,
    )

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(
        datagen.flow(x_train, y_train_2, batch_size=FLAGS.batch_size),
        validation_data=(x_test, y_test_2),
        epochs=epochs,
        initial_epoch=initial_epoch,
        verbose=1,
        workers=32,
        callbacks=callbacks,
    )
