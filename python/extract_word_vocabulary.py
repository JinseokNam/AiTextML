# -*- coding: utf-8 -*-
import numpy as np
import h5py
from utils import *
import sys
sys.path.append('/home/nam/codes/vectorlearn/python')

dataset_filepath = '/home/nam/codes/vectorlearn/preproc/2005_split_data/dataset.h5'

f = h5py.File(dataset_filepath,'r')

words_grp = f['Words']
words_freq_info_dset = words_grp['freq_info']
word_vocab_size = words_freq_info_dset.len()
word_vocab = np.array([words_freq_info_dset[i][0].decode('utf-8') for i in xrange(word_vocab_size)])

f.close()

for word in word_vocab:
  print '%s' % word.encode('utf-8')
