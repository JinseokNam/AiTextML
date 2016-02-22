# -*- coding: utf-8 -*-
import numpy as np
import h5py
from utils import *
import sys
sys.path.append('/home/nam/codes/vectorlearn/python')

dataset_filepath = '/home/nam/codes/vectorlearn/preproc/2005_split_data/dataset.h5'
model_vectors_filepath = '/home/nam/codes/vectorlearn/models/BioASQ_2005_split_test_model_alpha_033_beta_033_gamma_033.vectors'
inference_model_filepath = '/home/nam/codes/vectorlearn/models/BioASQ_2005_split_test_model_alpha_033_beta_033_gamma_033.inferred_testset_vectors'

f = h5py.File(dataset_filepath,'r')

labels_grp = f['Labels']
label_offset_dset = labels_grp['offset_info']
label_vocab_size = label_offset_dset.len()
label_vocab = np.array([label_offset_dset[i][0].decode('utf-8') for i in xrange(label_vocab_size)])
inv_label_vocab = {label_name:idx for idx, label_name in enumerate(label_vocab)}
f.close()

"""
Load the trained model and the test model
"""

model = load_model(model_vectors_filepath)
test_model = load_inferred_test_instances(inference_model_filepath)

seen_label_embeddings = model['label_emb']
unseen_label_embeddings = test_model['unseen_label_emb']
num_seen_labels = seen_label_embeddings.shape[0]

def nearest_neighbors(query_label,num_neighbors=10):
  print '=== %s ===' % query_label
  given_label_idx = inv_label_vocab[query_label]
  query = unseen_label_embeddings[given_label_idx-num_seen_labels,:]
  display_similar_labels(seen_label_embeddings,label_vocab[:num_seen_labels],query,k=num_neighbors)
  print '==='
  display_similar_labels(unseen_label_embeddings,label_vocab[num_seen_labels:],query,k=num_neighbors)

nearest_neighbors('Tundra')
nearest_neighbors('Hope')
nearest_neighbors('Night_Vision')
nearest_neighbors('Water_Resources')
nearest_neighbors('Upper_Extremity_Deep_Vein_Thrombosis')
nearest_neighbors('Data_Mining')
nearest_neighbors('Health_Information_Exchange')
nearest_neighbors('Sensorimotor_Cortex')
nearest_neighbors('Olfactory_Nerve_Diseases')
nearest_neighbors('Spinal_Cord_Dorsal_Horn')

