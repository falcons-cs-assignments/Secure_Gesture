import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer


# Siamese L1 Distance class
# this is a custom layer that we added to our siamese network to calc distance between features
# NOTE: make sure that you load this layer in custom objects if you Reload model weights
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

