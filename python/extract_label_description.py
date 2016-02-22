# -*- coding: utf-8 -*-
import numpy as np
import h5py
from utils import *
import sys
sys.path.append('/home/nam/codes/vectorlearn/python')

dataset_filepath = '/home/nam/codes/vectorlearn/preproc/2005_split_data/dataset.h5'

f = h5py.File(dataset_filepath,'r')

labels_grp = f['Labels']
label_offset_dset = labels_grp['offset_info']
label_vocab_size = label_offset_dset.len()
label_vocab = np.array([label_offset_dset[i][0].decode('utf-8') for i in xrange(label_vocab_size)])
inv_label_vocab = {label_name:idx for idx, label_name in enumerate(label_vocab)}
f.close()

for label_name in label_vocab:
  print '%s' % label_name
