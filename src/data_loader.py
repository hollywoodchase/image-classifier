import tensorflow as tf

def load_cifar10_data():
    return tf.keras.datasets.cifar10.load_data()
