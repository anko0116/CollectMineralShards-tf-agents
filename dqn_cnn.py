import tensorflow as tf
from tensorflow import keras
from tf_agents.agents.dqn import dqn_agent

def build_model():
    q_net = keras.Sequential()
    q_net.add(PreprocessLayer())
    q_net.add(tf.keras.layers.Conv2D(16, 3, strides=(3,3), padding='same', activation='relu'))
    q_net.add(tf.keras.layers.Conv2D(1, 1, strides=(1,1), padding='valid', activation='relu'))
    return q_net


class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(PreprocessLayer, self).__init__()
        self.marine_idx = 1
        self.mineral_idx = 3


    def build(self):
        pass
    def call(self, inputs):
        # https://www.tensorflow.org/api_docs/python/tf/math/equal
        flat_inputs = tf.reshape(inputs, (84*84))
        marine_indices = tf.math.equals(flat_inputs, 1) # Returns list of Trues and Falses
        mineral_indices = tf.math.equals(flat_inputs, 3)

        marine_img = tf.one_hot(marine_indices, 1)
        mineral_img = tf.one_hot(mineral_indices, 1)
        print(marine_img)
        print(mineral_img)
        return marine_img
            