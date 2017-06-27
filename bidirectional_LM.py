__author__ = 'Dimitris'



import numpy as np
import tensorflow as tf
import os
import cPickle as pickle
from pprint import pprint
import math
tf.reset_default_graph()
tf.set_random_seed(1989)
np.random.seed(1989)


print tf.__version__   # THIS SHOULD BE 1.1.0
print_every_n_batches = 500

import logging
logger = logging.getLogger('bidir_LM_GRU')
hdlr = logging.FileHandler('/home/dpappas/bidir_LM_GRU.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

def get_config():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    return config

def variable_summaries(var, namescope):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.device('/cpu:0'):
        with tf.name_scope(namescope):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

vocab_size  = 1500000
b_size      = 10
emb_size    = 100
num_units   = 100
timesteps   = 4

inputs  = tf.placeholder(tf.int32, shape=(None, timesteps))
outputs = tf.placeholder(tf.int32, shape=(None,))

embeddings  = tf.Variable( initial_value = tf.truncated_normal( shape = [ vocab_size, emb_size, ], stddev = 1.0 / math.sqrt(emb_size) ) , name='embeddings' )
variable_summaries(embeddings, 'embeddings')
nce_weights = tf.Variable( initial_value = tf.truncated_normal( shape = [ vocab_size, num_units ], stddev = 1.0 / math.sqrt(num_units) ) , name='nce_weights' )
variable_summaries(nce_weights, 'nce_weights')
nce_biases = tf.Variable( initial_value  = tf.random_uniform( shape = [ vocab_size ], minval = 0.1, maxval = 0.9 ) , name='nce_biases' )
variable_summaries(nce_biases, 'nce_biases')

embed = tf.nn.embedding_lookup(embeddings, inputs)
input = tf.unstack(tf.transpose(embed, [1, 0, 2]))

# grus = 0
# def create_gru_cell():
#     global grus
#     with tf.variable_scope("gru" + str(grus)):
#         grus += 1
#         ret = tf.contrib.rnn.GRUCell( num_units = num_units, input_size=None, activation=tf.tanh )
#         return ret
#
# cells_fw = 3 * [create_gru_cell()]
# # cells_fw = [tf.contrib.rnn.core_rnn_cell.MultiRNNCell(cells = cells_fw)]
# #stacked_lstm = tf.contrib.rnn.core_rnn_cell.MultiRNNCell(cells = cells)
# cells_bw = 3 * [create_gru_cell()]
# # cells_fw = [tf.contrib.rnn.core_rnn_cell.MultiRNNCell(cells = cells_bw)]
#
# outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_rnn(
#     cells_fw,
#     cells_bw,
#     input,
#     initial_states_fw=None,
#     initial_states_bw=None,
#     dtype=tf.float32,
#     sequence_length=None,
#     scope=None
# )

lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)

outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, input, dtype=tf.float32)





