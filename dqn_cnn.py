import tensorflow as tf
import numpy as np
from tensorflow import keras
from tf_agents.networks import sequential

import sys
#np.set_printoptions(threshold=sys.maxsize)

def build_model():
    layers = [
        PreprocessLayer(),
        tf.keras.layers.Conv2D(16, 3, strides=1, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(1, 1, strides=1, padding='same', activation=None), 
        PickQLayer(),
    ]
    q_net = sequential.Sequential(layers)
    return q_net


class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(PreprocessLayer, self).__init__()
        self.marine_idx = 1
        self.mineral_idx = 3


    def build(self, arg):
        self.marine_idx = 1
        self.mineral_idx = 3
    def call(self, inputs):
        # https://www.tensorflow.org/api_docs/python/tf/math/equal
        flat_inputs = tf.reshape(inputs, (84*84))
        marine_indices = tf.math.equal(flat_inputs, 1) # Returns list of Trues and Falses
        mineral_indices = tf.math.equal(flat_inputs, 3)

        marine_indices = tf.cast(marine_indices, tf.uint8)
        mineral_indices = tf.cast(mineral_indices, tf.uint8)

        marine_img = tf.one_hot(marine_indices, 1)
        mineral_img = tf.one_hot(mineral_indices, 1)
        #print(marine_img)
        #print(mineral_img)

        marine_img = tf.reshape(marine_img, (1, 84, 84, 1))
        mineral_img = tf.reshape(mineral_img, (1, 1, 84, 84))
        #print(marine_img)
        return marine_img

class PickQLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(PickQLayer, self).__init__()
    def build(self, arg):
        return
    def call(self, inputs):
        # Return location on the map with highest Q value
        #print(inputs)
        inputs = tf.reshape(inputs, (7056))
        idx = np.argmax(inputs)
        action = np.zeros((1, 7056))
        action[0][idx] = 1
        return action