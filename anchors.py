import os
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt


def get_face(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    print(file_path[0])
    print(file_path.shape)
    # HERE FACE DETECTION HAPPENS
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#FIXME get another cascade
    gray = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2GRAY)#FIXME raw file path
    (x, y, w, h) = face_cascade.detectMultiScale(gray, 1.1, 4)#FIXME gives a list
    # HERE FACE DETECTION HAPPENS
    img = img[y:y + h, x:x + w]
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return tf.convert_to_tensor(img)


def prepare_data(anchor_path, should_save=False):
    anchor = tf.data.Dataset.list_files(anchor_path + '\*.jpg').take(3000)
    anchor = anchor.map(get_face)

    img = anchor.as_numpy_iterator().next()
    plt.imshow(img)
    plt.show()

    anchor = anchor.cache()
    anchor = anchor.shuffle(buffer_size=10000)


prepare_data("test")
