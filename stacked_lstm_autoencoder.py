import numpy as np
import tensorflow as tf
from pprint import pprint
import math
tf.reset_default_graph()
tf.set_random_seed(1989)
np.random.seed(1989)

print tf.__version__   # THIS SHOULD BE 1.1.0
print_every_n_batches = 500

# vocab_size  = 1500000
# b_size      = 20
# timesteps   = 400
# emb_size    = 200
# num_units   = 300
# stack_size  = 3
# num_sampled = 10
# lr          = 1

vocab_size  = 100
b_size      = 6
timesteps   = 40
emb_size    = 10
num_units   = 20
stack_size  = 3
num_sampled = 10
lr          = 1

X = np.random.randint(vocab_size, size=(b_size, timesteps))
print X

# import logging
# logger = logging.getLogger('stacked_lstm_autoencoder')
# hdlr = logging.FileHandler('/home/dpappas/my_stacked_lstm_autoencoder.log')
# formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
# hdlr.setFormatter(formatter)
# logger.addHandler(hdlr)
# logger.setLevel(logging.INFO)

def get_config():
    config = tf.ConfigProto()
    config.allow_soft_placement=True
    config.log_device_placement = False
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    return config

class my_autoencoder(object):
    def __init__(
            self,
            b_size,
            timesteps,
            emb_size,
            vocab_size,
            num_units,
            stack_size,
            nce_num_sampled,
            learning_rate,
            mlp_size,
    ):
        self.timesteps = timesteps
        self.mlp_size = mlp_size
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.num_sampled = nce_num_sampled
        self.inputs = tf.placeholder(tf.int32, shape=(b_size, timesteps))
        self.lengths = tf.placeholder(tf.int32, shape=(b_size))
        self.lstms = 0
        self.num_units = num_units
        self.stack_size = stack_size
        self.learning_rate = learning_rate
        self.init_variables()
        with tf.device('/cpu:0'):
            self.embed  = tf.nn.embedding_lookup(self.embeddings, self.inputs)
        self.create_model_1()
    def init_variables(self):
        self.embeddings = tf.Variable(
            initial_value = tf.random_uniform(
                shape = [
                    self.vocab_size,
                    self.emb_size
                ],
                minval = -1.0,
                maxval = 1.0,
            )
        )
        self.global_step = tf.Variable(
            initial_value = 0,
            trainable     = False,
        )
        self.mlp_weights = tf.Variable(
            initial_value = tf.random_normal(
                shape = [
                    num_units,
                    self.mlp_size,
                ]
            )
        )
        self.bias = tf.Variable(
            initial_value = tf.random_normal(
                shape = [
                    self.mlp_size,
                ]
            )
        )
        self.nce_weights = tf.Variable(
            initial_value = tf.truncated_normal(
                shape = [
                    self.vocab_size,
                    self.num_units
                ],
                stddev = 1.0 / math.sqrt(self.num_units)
            )
        )
        self.nce_biases = tf.Variable(
            initial_value = tf.zeros(
                shape = [
                    self.vocab_size
                ]
            )
        )
    def create_lstm_cell(self):
        with tf.variable_scope("lstm" + str(self.lstms)):
            self.lstms += 1
            return tf.contrib.rnn.core_rnn_cell.LSTMCell(
                self.num_units,
                input_size=None,
                use_peepholes=False,
                cell_clip=None,
                initializer=None,
                num_proj=None,
                proj_clip=None,
                num_unit_shards=None,
                num_proj_shards=None,
                forget_bias=1.0,
                state_is_tuple=True,
                activation=tf.tanh,
            )
    def create_stack(self):
        cells = [
            tf.contrib.rnn.core_rnn_cell.DropoutWrapper(
                self.create_lstm_cell()
            ) for i in range(self.stack_size)
        ]
        stacked_lstm = tf.contrib.rnn.core_rnn_cell.MultiRNNCell(cells)
        return stacked_lstm
    def create_optimizer(self, opt, clip_by):
        if(opt.lower() == 'sgd'):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        if(clip_by == 'norm'):
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in gvs]
            self.optimizer = optimizer.apply_gradients(capped_gvs)
        if(clip_by == 'value'):
            gvs = optimizer.compute_gradients(self.loss)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.optimizer = optimizer.apply_gradients(capped_gvs)
        else:
            self.optimizer = optimizer.minimize(loss=self.loss, global_step=self.global_step)
    def create_model_1(self):
        with tf.device('/gpu:0'):
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                cell                = self.create_stack(),
                inputs              = self.embed,
                sequence_length     = self.lengths,
                initial_state       = None,
                dtype               = tf.float32,
                parallel_iterations = None,
            )
        with tf.device('/gpu:1'):
            def loop_f(prev, i):
                return prev
            enc_outputs = tf.transpose(self.encoder_outputs, [1, 0, 2])
            decode_input = [tf.zeros_like(enc_outputs[-1], dtype=tf.float32, name="GO") + enc_outputs[-1]] * self.timesteps
            self.decoder_outputs, self.decoder_state = tf.contrib.legacy_seq2seq.rnn_decoder(
                decoder_inputs  = decode_input,
                initial_state   = self.encoder_state,
                cell            = self.create_stack(),
                loop_function   = loop_f,
            )
        with tf.device('/gpu:0'):
            self.compute_loss()
        with tf.device('/cpu:0'):
            self.create_optimizer('sgd', None)
    def compute_loss(self):
        self.loss = tf.constant(0.0, dtype=tf.float32)
        inn = tf.reverse_sequence(
            input=self.inputs,
            seq_lengths=self.lengths,
            seq_axis=1,
            batch_axis=0,

        )
        inn = tf.transpose(inn, [1, 0])
        # inn = tf.transpose(self.inputs, [1, 0])
        for i in range(len(self.decoder_outputs)):
            l = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights     = self.nce_weights,
                    biases      = self.nce_biases,
                    labels      = tf.reshape(inn[i], [-1, 1]),
                    inputs      = self.decoder_outputs[i],
                    num_sampled = self.num_sampled,
                    num_classes = self.vocab_size,
                )
            )
            self.loss += l # / len(self.decoder_outputs)

ae = my_autoencoder(
    b_size          = b_size,
    timesteps       = timesteps,
    emb_size        = emb_size,
    vocab_size      = vocab_size,
    num_units       = num_units,
    stack_size      = stack_size,
    nce_num_sampled = num_sampled,
    learning_rate   = lr,
    mlp_size        = 100,
)

X = np.random.randint(vocab_size, size=(b_size, timesteps))
lens = (X != 0).sum(1)
# print X
sess = tf.Session(config=get_config())
sess.run(tf.global_variables_initializer())
for i in range(1000):
    _, l = sess.run(
        [
            ae.optimizer,
            ae.loss
        ],
        feed_dict={
            ae.inputs: X,
            ae.lengths: lens,
        },
    )
    print l
sess.close()


exit()




sess = tf.Session( config = config )
sess.run( tf.global_variables_initializer() )
saver = tf.train.Saver()
import os
import cPickle as pickle
p = '/home/dpappas/koutsouremeno_dataset/train/'
fs = os.listdir(p)

for epoch in range(20):
    sum_cost, m_batches = 0. , 0.
    for f in fs:
        m_batches+=1
        d = pickle.load(open(p+f,'rb'))
        _, l = sess.run(
            [
                ae.optimizer,
                ae.loss
            ],
            feed_dict = {
                ae.inputs  : d['context'],
                ae.lengths : (d['context'] != 0).sum(1),
            },
        )
        sum_cost += l
        # logger.info(
        #     'train b:{} e:{}. batch_cost is {}. average_cost is: {}.'.format(
        #         m_batches,
        #         epoch,
        #         '{0:.4f}'.format(l),
        #         '{0:.4f}'.format((sum_cost/(m_batches*1.0))),
        #     )
        # )
        print (
            'train b:{} e:{}. batch_cost is {}. average_cost is: {}.'.format(
                m_batches,
                epoch,
                '{0:.4f}'.format(l),
                '{0:.4f}'.format((sum_cost/(m_batches*1.0))),
            )
        )
        print(
            'train b:{} e:{}. batch_cost is {}. average_cost is: {}.'.format(
                m_batches,
                epoch,
                '{0:.4f}'.format(l),
                '{0:.4f}'.format((sum_cost/(m_batches*1.0))),
            )
        )
    save_path = saver.save(sess, './my_stacked_lstm_autoencoder_'+str(epoch)+'.ckpt')
    # logger.info('save_path: {}'.format( save_path ))
    print('save_path: {}'.format( save_path ))
    meta_graph_def = tf.train.export_meta_graph(filename = './my_limited_model_'+str(epoch)+'.meta')

sess.close()

'''

import pickle
import os

p = '/home/dpappas/koutsouremeno_dataset/train/'
fs = os.listdir(p)

maxx = 0
for f in fs:
    d = pickle.load(open(p+f,'rb'))
    print d['context'].max()
    break

d['context'].shape # 100, 500
d['quests'].shape
c_len = (d['context'] != 0).sum(1)
q_len = (d['quests'] != 0).sum(1)

'''
