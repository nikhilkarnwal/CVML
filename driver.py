import os

from ailabs.nlp import config
from ailabs.nlp.data import ptext
from ailabs.nlp import train

config.device = 'cpu'
config.dir = '../data'
path = lambda s: os.path.join(config.dir, s)
config.data_path = path('hin.txt')
config.vocab_path = path('vocab.json')
config.val_path = path('val_path.json')
config.resume = None

config.hidden_dim = 256
config.train = False
config.resume = path('2020-09-13_18_27_00_1_1_checkpoint.pth.tar')
config.eval = True

# ptext.execute(config.data_path, config.vocab_path)

config.epochs = 1
config.data_worker = 0
train.execute()

a = {}
