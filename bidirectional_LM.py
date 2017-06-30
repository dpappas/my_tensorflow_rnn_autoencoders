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

def manage_matrix(matrix):
    X = []
    Y = []
    for i in range(matrix.shape[0]):
        for j in range(timesteps, matrix.shape[1]):
            # print d['context'][i, j - timesteps:j], d['context'][i, j]
            X.append(matrix[i, j - timesteps:j])
            Y.append(matrix[i, j])
            if (matrix[i, j] == 0):
                break
    return X,Y

def yield_data(p,b_size):
    X = []
    Y = []
    fs = os.listdir(p)
    for f in fs:
        d = pickle.load(open(p+f,'rb'))
        Xt, Yt   = manage_matrix(d['context'])
        X.extend(Xt)
        Y.extend(Yt)
        Xt, Yt = manage_matrix(d['quests'])
        X.extend(Xt)
        Y.extend(Yt)
        while(len(X) > b_size):
            Xt = X[:b_size]
            Yt = Y[:b_size]
            Xt = np.array(Xt)
            Yt = np.array(Yt)
            Yt = Yt.reshape(Yt.shape[0], 1)
            yield Xt, Yt
            X  = X[b_size:]
            Y  = Y[b_size:]

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

vocab_size      = 1500000
b_size          = 2000
emb_size        = 100
num_units       = 100
timesteps       = 5
learning_rate   = 1.0
num_sampled     = 64

inputs  = tf.placeholder(tf.int32, shape=(None, timesteps))
outputs = tf.placeholder(tf.int32, shape=(None))

embeddings  = tf.Variable( initial_value = tf.truncated_normal( shape = [ vocab_size, emb_size, ], stddev = 1.0 / math.sqrt(emb_size) ) , name='embeddings' )
embed = tf.nn.embedding_lookup(embeddings, inputs)
input = tf.unstack(tf.transpose(embed, [1, 0, 2]))

gru_fw_cell = tf.contrib.rnn.GRUCell( num_units = num_units, input_size=None, activation=tf.tanh )
gru_bw_cell = tf.contrib.rnn.GRUCell( num_units = num_units, input_size=None, activation=tf.tanh )

bi_outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn(
    gru_fw_cell,
    gru_bw_cell,
    input,
    dtype=tf.float32
)

# print len(bi_outputs)
# print bi_outputs[0].get_shape()

# variable_summaries(embeddings, 'embeddings')
# variable_summaries(nce_weights, 'nce_weights')
# variable_summaries(nce_biases, 'nce_biases')
# variable_summaries(bi_outputs, 'bidirectional_outputs')
# variable_summaries(output_state_fw, 'fw_state')
# variable_summaries(output_state_bw, 'bw_state')

# o_weights = tf.Variable( initial_value = tf.truncated_normal( shape = [ 2*num_units, vocab_size ], stddev = 1.0 / math.sqrt(num_units) ) , name='o_weights' )
# o_biases = tf.Variable( initial_value  = tf.random_uniform( shape = [ vocab_size ], minval = 0.1, maxval = 0.9 ) , name='o_biases' )
# logits = tf.add(tf.matmul(bi_outputs[-1], o_weights), o_biases)
# loss =  tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits( labels = tf.contrib.layers.one_hot_encoding(outputs, vocab_size), logits = logits, ) )

weights = tf.Variable( initial_value = tf.truncated_normal( shape = [ vocab_size, 2*num_units], stddev = 1.0 / math.sqrt(num_units) ) , name='o_weights' )
biases = tf.Variable( initial_value  = tf.random_uniform( shape = [ vocab_size ], minval = 0.1, maxval = 0.9 ) , name='o_biases' )


mode = 'train'
if mode == "train":
  loss = tf.reduce_mean(
      tf.nn.sampled_softmax_loss(
        weights                 = weights,
        biases                  = biases,
        labels                  = outputs,
        inputs                  = bi_outputs[-1],
        num_sampled             = num_sampled,
        num_classes             = vocab_size,
        num_true                = 1,
        sampled_values          = None,
        remove_accidental_hits  = True,
        partition_strategy      = 'mod',
        name                    = 'sampled_softmax_loss'
      )
  )
elif mode == "eval":
  logits = tf.matmul(inputs, tf.transpose(weights))
  logits = tf.nn.bias_add(logits, biases)
  labels_one_hot = tf.one_hot(outputs, vocab_size)
  loss = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels_one_hot,
      logits=logits
  )
else:
    loss = None

# train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

train_op = tf.train.AdagradOptimizer( learning_rate, initial_accumulator_value=0.1, use_locking=False, name='Adagrad' ).minimize(loss)

yie = yield_data('/media/dpappas/dpappas_data/biomedical/more_koutsouremeno_dataset/train/',b_size)

sess    = tf.Session(config=get_config())
# merge_summary = tf.summary.merge_all()
# writer  = tf.summary.FileWriter('/tmp/teacher_stacked/1')
# writer.add_graph(sess.graph)
sess.run(tf.global_variables_initializer())

for epoch in range(10):
    for xt,yt in yie:
        _, l = sess.run(
            [ train_op, loss ],
            feed_dict={ inputs  : xt, outputs : yt, },
        )
        print epoch,l

sess.close()




# X = np.random.randint(vocab_size, size=(b_size, timesteps))
# Y = np.random.randint(vocab_size, size=(b_size, 1))
#
# sess    = tf.Session(config=get_config())
# # merge_summary = tf.summary.merge_all()
# # writer  = tf.summary.FileWriter('/tmp/teacher_stacked/1')
# # writer.add_graph(sess.graph)
# sess.run(tf.global_variables_initializer())
#
# for i in range(1000):
#     _, l = sess.run(
#         [ train_op, loss ],
#         feed_dict={ inputs  : X, outputs : Y, },
#     )
#     print l
#
# sess.close()


#
# def all_data(p):
#     X = []
#     Y = []
#     fs = os.listdir(p)
#     m=0
#     for f in fs:
#         d = pickle.load(open(p+f,'rb'))
#         Xt, Yt   = manage_matrix(d['context'])
#         X.extend(Xt)
#         Y.extend(Yt)
#         Xt, Yt = manage_matrix(d['quests'])
#         X.extend(Xt)
#         Y.extend(Yt)
#         m+=1
#         print 'finished {} of {}.'.format(m,len(fs))
#     X = np.array(X)
#     Y = np.array(Y)
#     Y = Y.reshape(Y.shape[0], 1)
#     return X,Y
#
# yie = all_data('/media/dpappas/dpappas_data/biomedical/more_koutsouremeno_dataset/train/')
# pickle.dump(yie[0], open('/media/dpappas/dpappas_data/biomedical/more_koutsouremeno_dataset/ngrams_X.p','wb'))
# pickle.dump(yie[1], open('/media/dpappas/dpappas_data/biomedical/more_koutsouremeno_dataset/ngrams_Y.p','wb'))

