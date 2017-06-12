import numpy as np
import tensorflow as tf
from pprint import pprint
import math
tf.reset_default_graph()

print tf.__version__   # THIS SHOULD BE 1.1.0

# vocab_size  = 100
# b_size      = 60
# timesteps   = 400
# emb_size    = 200
# num_units   = 300
# stack_size  = 5
# num_sampled = 10
# lr          = 1

vocab_size  = 100
b_size      = 6
timesteps   = 4
emb_size    = 10
num_units   = 20
stack_size  = 5
num_sampled = 10
lr          = 1
X = np.random.randint(vocab_size, size=(b_size, timesteps))
print X


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
        self.timesteps = timesteps
        self.vocab_size = vocab_size
        self.num_sampled = nce_num_sampled
        self.inputs = tf.placeholder(tf.int32, shape=(b_size, timesteps))
        self.embeddings = tf.Variable(tf.random_uniform([vocab_size, emb_size], -1.0, 1.0))
        self.embed  = tf.nn.embedding_lookup(self.embeddings, self.inputs)
        self.lstms = 0
        self.num_units = num_units
        self.stack_size = stack_size
        self.learning_rate = learning_rate
        self.create_model_1()
    def create_lstm_cell(self):
        self.lstms
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
            self.global_step = tf.Variable(0, trainable=False)
            self.optimizer = optimizer.minimize(loss=self.loss, global_step=self.global_step)
    def create_model_1(self):
        self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
            cell                = self.create_stack(),
            inputs              = self.embed,
            sequence_length     = None,
            initial_state       = None,
            dtype               = tf.float32,
            parallel_iterations = None,
        )
        def loop_f(prev, i):
            return prev
        enc_outputs = tf.transpose(self.encoder_outputs, [1, 0, 2])
        decode_input = [tf.zeros_like(enc_outputs[-1], dtype=np.float32, name="GO") + enc_outputs[-1]] * self.timesteps
        self.decoder_outputs, self.decoder_state = tf.contrib.legacy_seq2seq.rnn_decoder(
            decoder_inputs=decode_input,
            initial_state=self.encoder_state,
            cell=self.create_stack(),
            loop_function=loop_f,
        )
        self.compute_loss()
        self.create_optimizer('sgd', None)
    def compute_loss(self):
        self.nce_weights = tf.Variable( tf.truncated_normal([self.vocab_size, self.num_units], stddev=1.0 / math.sqrt(self.num_units)))
        self.nce_biases = tf.Variable(tf.zeros([self.vocab_size]))
        self.loss = tf.constant(0.0, dtype=tf.float32)
        inn = tf.transpose(self.inputs, [1, 0])
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
)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, l = sess.run(
        [
            ae.optimizer,
            ae.loss
        ],
        feed_dict = {
            ae.inputs : X
        },
    )
    print l

sess.close()



