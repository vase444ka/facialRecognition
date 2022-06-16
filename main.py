import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import random

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.python.keras.optimizer_v2.adam import Adam

import tensorflow as tf

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')


class CustomModel(Model):
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            X = batch[:2]
            y = batch[2]

            yhat = self(X, training=True)
            loss = self.compiled_loss(y, yhat, regularization_losses=self.losses)
        print(loss)#we'll see about this

        # Calculate gradients
        grad = tape.gradient(loss, self.trainable_variables)

        # Calculate updated weights and apply to siamese model
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, yhat)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0

    return img


def preprocess_twin(input_img, validation_img, label):
    return preprocess(input_img), preprocess(validation_img), label


def build_convolution():
    inp = Input(shape=(100, 100, 3), name='input_image')
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    return CustomModel(inputs=[inp], outputs=[d1], name='embedding')


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def build_siamese(internal):
    input_image = Input(name='input_img', shape=(100, 100, 3))
    validation_image = Input(name='validation_img', shape=(100, 100, 3))

    inp_convolution = internal(input_image)
    val_convolution = internal(validation_image)

    siamese_layer = L1Dist()
    distances = siamese_layer(inp_convolution, val_convolution)
    classifier = Dense(1, activation='sigmoid')(distances)
    return CustomModel(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


if __name__ == '__main__':
    anchor = tf.data.Dataset.list_files(ANC_PATH + '\*.jpg').take(3000)
    positive = tf.data.Dataset.list_files(POS_PATH + '\*.jpg').take(3000)
    negative = tf.data.Dataset.list_files(NEG_PATH + '\*.jpg').take(3000)

    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positives.concatenate(negatives)

    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=10000)

    train_data = data.take(round(len(data) * .7))
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)

    test_data = data.skip(round(len(data) * .7))
    test_data = test_data.take(round(len(data) * .3))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)

    siamese_network = build_siamese(build_convolution())
    print(siamese_network.summary())

    siamese_network.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=tf.losses.BinaryCrossentropy(),
        metrics=["mae"]
    )
    siamese_network.fit(train_data, epochs=10)
    siamese_network.save('siamesemodel.h5')
    #checkpoints??