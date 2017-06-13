import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
try:
    import cPickle as pickle
except:
    import pickle
import elasticsearch
from elasticsearch.helpers import scan
from pprint import pprint
np.random.seed(1989)



class instanciator(object):
    def __init__(
        self,
        vocab_path,
        elasticsearch_ip,
        index_name,
        max_len = 400,
        min_len = 100,
    ):
        self.max_len = max_len
        self.min_len = min_len
        self.index = index_name
        self.client = elasticsearch.Elasticsearch(elasticsearch_ip, verify_certs=True)
        self.vocab_path = vocab_path
        self.load_vocab()
    def load_vocab(self):
        vocab = pickle.load(open(self.vocab_path,'rb'))
        vocab = ['PAD', 'UNKN', '<EOS>', '<SOS>'] + vocab
        self.vocab = dict([ (elem[1],elem[0]) for elem in enumerate(vocab) ])
        self.inv_vocab = dict([ (elem[0],elem[1]) for elem in enumerate(vocab) ])
    def split_sentences(self, text):
        sents = sent_tokenize(text)
        ret = []
        i = 0
        while (i < len(sents)):
            sent = sents[i]
            while (
                sent.lower().endswith('fig.') or
                sent.lower().endswith('e.g.') or
                sent.lower().endswith('etc.') or
                sent.lower().endswith('et al.') or
                sent.lower().endswith(' cf.') or
                sent.lower().endswith('(cf.')
            ):
                sent += sents[i + 1]
                i += 1
            ret.append(sent)
            i += 1
        return ret
    def tokenize(self, text):
        return word_tokenize(text.lower())
    def yield_data(self):
        scroll = scan(self.client, index=self.index, query=None,
                      scroll=u'5m', raise_on_error=True,
                      preserve_order=False, size=1000,
                      request_timeout=None, clear_scroll=True)
        for res in scroll:
            tokens = self.tokenize(res['_source']['AbstractText'])
            if(len(tokens)< self.max_len and len(tokens) > self.min_len ):
                t = []
                for token in tokens:
                    try:
                        t.append(self.vocab[token])
                    except:
                        t.append(self.vocab['UNKN'])
                t = t + ([self.vocab['PAD']] * (self.max_len - len(t)) )
                yield t, len(tokens)


inst = instanciator(
    vocab_path          = '/home/dpappas/my_bio_vocab.p',
    elasticsearch_ip    = '',
    index_name          = '',
)


yielder = inst.yield_data()
for t in yielder:
    print t[1] , t[0]






