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
from nltk import word_tokenize
from nltk import sent_tokenize

def check_numeric(w):
    try:
        r = int(w.replace(',',''))
        return 'INTEGER'
    except ValueError:
        try:
            r = float(w.replace(',',''))
            return 'FLOAT'
        except ValueError:
            return w

def handle_new_text(text):
    sents = sent_tokenize(text)
    tokens = []
    for sent in sents:
        t = [ check_numeric(w) for w in word_tokenize(sent) ]
        tokens.append('START')
        tokens.extend(t)
        tokens.append('END')
    ind = []
    for token in tokens:
        try:
            ind.append(v_dict[token])
        except:
            ind.append(v_dict['UNKN'])
    return ind

def get_ngrams(ind, n):
    X, Y = [], []
    for i in range( len(ind)-n):
        X.append(ind[i:i+n])
        Y.append(ind[i+n])
    return X, Y

def yield_train_data(b_size, timesteps):
    p = '/home/DATA/pubmed_open_data/pmc_per_year/2014/'
    fs = [ p+f for f in os.listdir(p)]
    p = '/home/DATA/pubmed_open_data/pmc_per_year/2015/'
    fs.extend([ p+f for f in os.listdir(p)])
    p = '/home/DATA/pubmed_open_data/pmc_per_year/2016/'
    fs.extend([ p+f for f in os.listdir(p)])
    X,Y = [],[]
    m = 0
    for file in fs:
        t = open(file,'r').read().lower().replace('\n',' ').decode('utf-8')
        t = get_ngrams(handle_new_text(t), timesteps)
        X.extend(t[0])
        Y.extend(t[1])
        while(len(X)>b_size):
            x_ = np.array(X[:b_size])
            y_ = np.array(Y[:b_size])
            y_ = y_.reshape(y_.shape[0], 1)
            yield x_, y_
            X = X[b_size:]
            Y = Y[b_size:]
        m+=1
        print 'file {} of {}'.format(m, len(fs))

def yield_valid_data(b_size, timesteps):
    p = '/home/DATA/pubmed_open_data/pmc_per_year/2008/'
    fs = [ p+f for f in os.listdir(p)]
    X,Y = [],[]
    m = 0
    for file in fs:
        t = open(file,'r').read().lower().replace('\n',' ').decode('utf-8')
        t = get_ngrams(handle_new_text(t), timesteps)
        X.extend(t[0])
        Y.extend(t[1])
        while(len(X)>b_size):
            x_ = np.array(X[:b_size])
            y_ = np.array(Y[:b_size])
            y_ = y_.reshape(y_.shape[0], 1)
            yield x_, y_
            X = X[b_size:]
            Y = Y[b_size:]
        m+=1
        print 'file {} of {}'.format(m, len(fs))

def get_config():
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.log_device_placement = False
    config.gpu_options.allow_growth = False
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    return config

def create_model():
    inputs  = tf.placeholder(tf.int32, shape=(None, timesteps))
    outputs = tf.placeholder(tf.int32, shape=(None))
    #
    with tf.device('/cpu:0'):
        embeddings  = tf.Variable( initial_value = tf.truncated_normal( shape = [ vocab_size, emb_size, ], stddev = 1.0 / math.sqrt(emb_size) ) , name='embeddings' )
        embed = tf.nn.embedding_lookup(embeddings, inputs)
        embed = tf.contrib.layers.batch_norm(embed, center=True, scale=True, is_training=True )
        input = tf.unstack(tf.transpose(embed, [1, 0, 2]))
    #
    with tf.device('/gpu:0'):
        gru_fw_cell = tf.contrib.rnn.GRUCell( num_units = num_units, input_size=None, activation=tf.tanh )
        gru_bw_cell = tf.contrib.rnn.GRUCell( num_units = num_units, input_size=None, activation=tf.tanh )
        bi_outputs, output_state_fw, output_state_bw = tf.contrib.rnn.static_bidirectional_rnn( gru_fw_cell, gru_bw_cell, input, dtype=tf.float32 )
        bi_outputs = tf.contrib.layers.batch_norm(bi_outputs, center=True, scale=True, is_training=True)
    #
    with tf.device('/cpu:0'):
        weights = tf.Variable( initial_value = tf.truncated_normal( shape = [ vocab_size, 2*num_units], stddev = 1.0 / math.sqrt(num_units) ) , name='o_weights' )
        biases = tf.Variable( initial_value  = tf.random_uniform( shape = [ vocab_size ], minval = 0.1, maxval = 0.9 ) , name='o_biases' )
    #
    # variable_summaries(embeddings, 'embeddings')
    # variable_summaries(weights, 'weights')
    # variable_summaries(biases, 'biases')
    # variable_summaries(bi_outputs, 'bidirectional_outputs')
    # variable_summaries(output_state_fw, 'fw_state')
    # variable_summaries(output_state_bw, 'bw_state')
    #
    with tf.device('/cpu:0'):
        mode = 'train'
        if mode == "train":
            loss = tf.reduce_mean( tf.nn.sampled_softmax_loss( weights = weights, biases = biases, labels = outputs, inputs = bi_outputs[-1], num_sampled = num_sampled, num_classes = vocab_size, num_true = 1, sampled_values = None, remove_accidental_hits = True, partition_strategy = 'mod', name = 'sampled_softmax_loss' ) )
        elif mode == "eval":
            logits = tf.matmul(inputs, tf.transpose(weights))
            logits = tf.nn.bias_add(logits, biases)
            labels_one_hot = tf.one_hot(outputs, vocab_size)
            loss = tf.nn.softmax_cross_entropy_with_logits( labels=labels_one_hot, logits=logits )
        else:
            loss = None
        #
        train_op = tf.train.AdagradOptimizer( learning_rate, initial_accumulator_value=0.1, use_locking=False, name='Adagrad' ).minimize(loss)
        # train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    #
    return inputs, outputs, loss, train_op

vocab = pickle.load(open('/home/dpappas/my_pmc_vocab.p','rb'))
vocab.append('START')
vocab.append('END')

v_dict = {}
for i in range(len(vocab)):
    v_dict[vocab[i]] = i


print tf.__version__   # THIS SHOULD BE 1.1.0
print_every_n_batches = 500

import logging
logger = logging.getLogger('bidir_LM_GRU')
hdlr = logging.FileHandler('/home/dpappas/milion_bidir_LM_GRU.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)


vocab_size      = 3500000
b_size          = 10000
emb_size        = 100
num_units       = 200
timesteps       = 5
learning_rate   = 1.00
num_sampled     = 64

inputs, outputs, loss, train_op = create_model()

sess = tf.Session(config=get_config())
merge_summary = tf.summary.merge_all()
loss_summary = tf.summary.scalar('loss',loss)
train_writer = tf.summary.FileWriter('/tmp/million_bidirectional_LM/train')
valid_writer = tf.summary.FileWriter('/tmp/million_bidirectional_LM/valid')
train_writer.add_graph(sess.graph)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

test_average_loss = tf.Variable(0.0)
test_average_loss_summary = tf.summary.scalar('loss',test_average_loss)

i = 0
i2 = 0

for epoch in range(10):
    yie = yield_train_data(b_size, timesteps)
    sum_cost, m_batches = 0. , 0.
    for xt,yt in yie:
        m_batches+=1
        _, l, ls = sess.run( [ train_op, loss, loss_summary], feed_dict={ inputs  : xt, outputs : yt, }, )
        train_writer.add_summary(ls, i)
        sum_cost += l
        print( 'train b:{} e:{}. batch_cost is {}. average_cost is: {}.'.format( m_batches, epoch, '{0:.4f}'.format(l), '{0:.4f}'.format((sum_cost/(m_batches*1.0))), ) )
        logger.info('train b:{} e:{}. batch_cost is {}. average_cost is: {}.'.format( m_batches, epoch, '{0:.4f}'.format(l), '{0:.4f}'.format((sum_cost/(m_batches*1.0))), ) )
        i+=1
    # validation time
    yie_valid = yield_valid_data(b_size, timesteps)
    sum_cost, m_batches = 0. , 0.
    for xt,yt in yie_valid:
        m_batches+=1
        l = sess.run( loss, feed_dict={ inputs  : xt, outputs : yt, }, )
        sum_cost += l
    # assign the average test loss to a variable
    assign_op = test_average_loss.assign((sum_cost/(m_batches*1.0)))
    sess.run(assign_op)
    ls = sess.run( test_average_loss_summary )
    valid_writer.add_summary(ls, i)
    #
    print( 'valid b:{} e:{}. average_cost is: {}.'.format( m_batches, epoch, '{0:.4f}'.format((sum_cost/(m_batches*1.0))), ) )
    logger.info( 'valid b:{} e:{}. average_cost is: {}.'.format( m_batches, epoch, '{0:.4f}'.format((sum_cost/(m_batches*1.0))), ) )
    # saving time
    save_path = saver.save(sess, './my_million_bidirectional_LM_model_'+str(epoch)+'.ckpt')
    # logger.info('save_path: {}'.format( save_path ))
    print('save_path: {}'.format( save_path ))
    logger.info('save_path: {}'.format( save_path ))
    meta_graph_def = tf.train.export_meta_graph(filename = './my_million_bidirectional_LM_model_'+str(epoch)+'.meta')




# X = np.random.randint(vocab_size, size=(b_size, timesteps))
# Y = np.random.randint(vocab_size, size=(b_size, 1))


