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

vocab_size  = 1500000
b_size      = 400
timesteps   = 400
emb_size    = 300
num_units   = 1500
stack_size  = 1
num_sampled = 100
lr          = 1

# import logging
# logger = logging.getLogger('stacked_lstm_autoencoder')
# hdlr = logging.FileHandler('/home/dpappas/my_stacked_lstm_autoencoder.log')
# formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
# hdlr.setFormatter(formatter)
# logger.addHandler(hdlr)
# logger.setLevel(logging.INFO)

def get_config():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
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
    ):
        self.b_size         = b_size
        self.emb_size       = emb_size
        self.timesteps      = timesteps
        self.vocab_size     = vocab_size
        self.num_sampled    = nce_num_sampled
        self.lstms          = 0
        self.num_units      = num_units
        self.stack_size     = stack_size
        self.learning_rate  = learning_rate
        self.inputs         = tf.placeholder(tf.int32, shape=(b_size, timesteps))
        self.lengths        = tf.placeholder(tf.int32, shape=(b_size))
        self.init_variables()
        with tf.device('/cpu:0'):
            self.embed      = tf.nn.embedding_lookup(self.embeddings, self.inputs)
            # print self.embed.get_shape()
        self.create_model_1()
    def init_variables(self):
        self.global_step = tf.Variable(
            initial_value = 0,
            trainable     = False,
        )
        self.go = tf.Variable(
            initial_value=tf.random_uniform(
                shape=[ self.emb_size, ],
                minval=-1.0,
                maxval=1.0,
            ),
            dtype=tf.float32,
            name="GO",
        )
        self.go = tf.stack(b_size * [self.go])
        # print self.go.get_shape()
        with tf.device('/cpu:0'):
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
                num_units = self.num_units,
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
                cell = self.create_lstm_cell()
            ) for i in range(self.stack_size)
        ]
        stacked_lstm = tf.contrib.rnn.core_rnn_cell.MultiRNNCell(cells = cells)
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
        with tf.device('/gpu:0'):
            decode_input = [
                self.go
            ] + tf.unstack(tf.transpose(self.embed, [1, 0, 2]))[:-1]
            # pprint(decode_input)
            self.decoder_outputs, self.decoder_state = tf.contrib.legacy_seq2seq.rnn_decoder(
                decoder_inputs  = decode_input,
                initial_state   = self.encoder_state,
                cell            = self.create_stack(),
                loop_function   = None,
            )
        with tf.device('/gpu:0'):
            self.compute_loss()
        with tf.device('/cpu:0'):
            self.create_optimizer('sgd', 'norm')
    def body(self,i, lllll):
        l = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=self.nce_weights,
                biases=self.nce_biases,
                labels=tf.reshape(
                    tf.gather(
                        self.inn,
                        i
                    ),
                    [-1, 1]
                ),
                inputs=tf.gather(
                    self.decoder_outputs,
                    i,
                ),
                num_sampled=self.num_sampled,
                num_classes=self.vocab_size,
            )
        )
        lllll += l
        return [tf.add(i, 1), lllll]
    def compute_loss_with_while(self):
        self.loss = tf.constant(0.0, dtype=tf.float32)
        self.inn = tf.reverse_sequence(
            input       = self.inputs,
            seq_lengths = self.lengths,
            seq_axis    = 1,
            batch_axis  = 0,
        )
        self.inn = tf.transpose(self.inn, [1, 0])
        print self.inn.get_shape()
        max_len = tf.reduce_max( input_tensor = self.lengths , axis = None, keep_dims = False, name = None, reduction_indices=None, )
        i = tf.constant(0)
        while_condition = lambda i, lllll : tf.less(i, max_len)
        self.decoder_outputs = tf.stack(self.decoder_outputs)
        i, self.loss, = tf.while_loop( while_condition, self.body, [ i, self.loss, ] )
    def compute_loss(self):
        self.loss = tf.constant(0.0, dtype=tf.float32)
        inn = tf.reverse_sequence(
            input       = self.inputs,
            seq_lengths = self.lengths,
            seq_axis    = 1,
            batch_axis  = 0,
        )
        inn = tf.transpose(inn, [1, 0])
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
            self.loss += l

ae = my_autoencoder(
    b_size          = b_size,
    timesteps       = timesteps,
    emb_size        = emb_size,
    vocab_size      = vocab_size,
    num_units       = num_units,
    stack_size      = stack_size,
    nce_num_sampled = num_sampled,
    learning_rate   = lr,
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


sess = tf.Session(config=get_config())
sess.run(tf.global_variables_initializer())
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



# the predicate for stopping the while loop. Tensorflow demands that we have
# all of the variables used in the while loop in the predicate.
pred = lambda prob,counter,state,input,acc_states,acc_output,acc_probs:\
    tf.logical_and(tf.less(prob,self.one_minus_eps), tf.less(counter,self.N))

def ACTStep(self,prob, counter, state, input, acc_states, acc_outputs, acc_probs):
    #
    #run rnn once
    output, new_state = rnn.rnn(self.cell, [input], state, scope=type(self.cell).__name__)
    #
    prob_w = tf.get_variable("prob_w", [self.cell.input_size,1]) 
    prob_b = tf.get_variable("prob_b", [1])
    halting_probability = tf.nn.sigmoid(tf.matmul(prob_w,new_state) + prob_b) 
    #
    acc_states.append(new_state)
    acc_outputs.append(output)
    acc_probs.append(halting_probability) 
    #
    return [p + prob, counter + 1.0, new_state, input,acc_states,acc_outputs,acc_probs]




'''
