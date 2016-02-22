#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import h5py
from utils import *
import time
import sys
sys.path.append('/home/nam/codes/vectorlearn/python')

dataset_filepath = '/home/nam/codes/vectorlearn/preproc/2005_split_data/dataset.h5'
model_vectors_filepath = '/home/nam/codes/vectorlearn/models/BioASQ_2005_split_test_model_alpha_033_beta_033_gamma_033.vectors.latest'
inference_model_filepath = '/home/nam/codes/vectorlearn/models/BioASQ_2005_split_test_model_alpha_033_beta_033_gamma_033.latest.inferred_testset_vectors'
#model_vectors_filepath = '/home/nam/codes/vectorlearn/models/BioASQ_2005_split_test_model_alpha_033_beta_066_gamma_0.vectors'
#inference_model_filepath = '/home/nam/codes/vectorlearn/models/BioASQ_2005_split_test_model_alpha_033_beta_066_gamma_0.inferred_testset_vectors'

f = h5py.File(dataset_filepath,'r')

split_grp = f['Split']
num_test_instances = split_grp['Testset'].attrs['num_instances']
base_offset = split_grp['Testset'][0][1] # PMID, offset, dummy

doc_label_pairs_grp = f['Document_Label_Pair']
offset_info_dset = np.array(doc_label_pairs_grp['offset_info'])
labels_seq_dset = np.array(doc_label_pairs_grp['entity_indices'])

model = load_model(model_vectors_filepath)
train_doc_representations = model['doc_emb']
num_train_instances = train_doc_representations.shape[0]
seen_label_embeddings = model['label_emb']
L=seen_label_embeddings.shape[0]
trans_mat = np.transpose(model['trans_mat'])

test_model = load_inferred_test_instances(inference_model_filepath)
test_doc_representations = test_model['test_doc_emb']

num_active_test_instances = 0

with open('label_desc_rankloss_tracking_latest.txt','w') as fout, open('label_desc_ranking_latest.txt','w') as fout_rank:
  rankloss_list = num_test_instances*[None]
  #rankloss_list = num_train_instances*[None]
  for inst_idx in xrange(num_test_instances):
  #for inst_idx in xrange(num_train_instances):
    selected_instance_offset = offset_info_dset[base_offset+inst_idx]
    label_offset = selected_instance_offset[1]
    num_labels = selected_instance_offset[2]

    mapped_test_inst_reprs = np.dot(trans_mat,test_doc_representations[inst_idx,:]) # projection of instance representations
    #mapped_train_inst_reprs = np.dot(trans_mat,train_doc_representations[inst_idx,:]) # projection of instance representations

    #sim_scores = np.sum(seen_label_embeddings*mapped_test_inst_reprs,axis=1)  # computing similarity scores
    sim_scores = np.dot(seen_label_embeddings,mapped_test_inst_reprs)  # computing similarity scores
    #sim_scores = np.sum(seen_label_embeddings*mapped_train_inst_reprs,axis=1)  # computing similarity scores

    pos_labels = [label for label in labels_seq_dset[label_offset:label_offset+num_labels] if label < L]

    if len(pos_labels) > 0:
      num_active_test_instances += 1
      p_rankloss, pos_label_ranks = eval_rankloss(sim_scores, pos_labels)
      fout.write('%.5f\n' % p_rankloss)
      fout_rank.write('%s\n' % ','.join([str(lid)+':'+str(rank) for lid, rank in pos_label_ranks]))
    else:
      p_rankloss = 0
      fout.write('#no_positive_labels\n')
      fout_rank.write('#no_positive_labels\n')

    fout.flush()
    fout_rank.flush()

    rankloss_list[inst_idx] = p_rankloss
    print '%.2f%%\r' % (((inst_idx+1)/float(num_test_instances))*100),
    #print '%.2f%%\r' % (((inst_idx+1)/float(num_train_instances))*100),

print np.sum(np.array(rankloss_list))/float(num_active_test_instances)
  
