# -*- coding: utf-8 -*-
import h5py
import numpy as np
import sys 

# construct dictionary {label_size: [label_idx_1, label_idx_2, ..., label_idx_\phi(label_size)]}
dataset_filepath = '/home/nam/codes/vectorlearn/preproc/2005_split_data/dataset.h5'
#path_to_test_predictions = '/home/nam/codes/vectorlearn/python/no_label_desc_ranking.txt'
#path_to_output_lmrr = '/home/nam/codes/vectorlearn/python/lMRR_no_label_desc.txt'
path_to_test_predictions = '/home/nam/codes/vectorlearn/python/label_desc_ranking_latest.txt'
path_to_output_lmrr = '/home/nam/codes/vectorlearn/python/lMRR_label_desc_latest.txt'

f = h5py.File(dataset_filepath,'r')

split_grp = f['Split']
num_train_instances = split_grp['Trainset'].attrs['num_instances']
num_test_instances = split_grp['Testset'].attrs['num_instances']

labels_grp = f['Labels']
label_offset_dset = labels_grp['offset_info']
label_vocab_size = label_offset_dset.len()
label_vocab = np.array([label_offset_dset[i][0].decode('utf-8') for i in xrange(label_vocab_size)])
inv_label_vocab = {label_name:int(idx) for idx, label_name in enumerate(label_vocab)}

doc_label_pairs_grp = f['Document_Label_Pair']
offset_info_dset = np.array(doc_label_pairs_grp['offset_info'])
labels_seq_dset = np.array(doc_label_pairs_grp['entity_indices'])

f.close()

print 'Loaded the dataset'

# Assume that we have loaded training label patterns
label_id_size_pairs = dict()    # This corresponds to the label size (Number of documents) {label_id : label_size}
base_offset = 0 
for inst_idx in xrange(num_train_instances):
  selected_instance_offset = offset_info_dset[base_offset+inst_idx]
  label_offset = selected_instance_offset[1]
  num_labels = selected_instance_offset[2]
  labels = [int(label) for label in labels_seq_dset[label_offset:label_offset+num_labels]]
  for label in labels:
    if not label_id_size_pairs.has_key(label):
      label_id_size_pairs[label] = 0 
    label_id_size_pairs[label] += 1

print 'Loaded training label patterns'
print 'Now building a dictionary for the label id and its size'

with open(path_to_test_predictions) as fin:
  test_predictions = []
  for line in fin:
    label_rank_pairs = []
    if line.startswith('#'):
      continue
    for label_rank_pair in line.strip().split(','):
      label,rank = label_rank_pair.split(':')
      label_rank_pairs.append((int(label),int(rank)))
    test_predictions.append(label_rank_pairs)

# construct dictionary {label_size : [doc_id_1, doc_id_2, ..., doc_id_N]}
"""
    The following structure corresponds to the set \mathcal{A}_s in the paper
"""
label_sets_dict = {}
for doc_idx,label_rank_pairs in enumerate(test_predictions):
  for label,rank in label_rank_pairs:
    label_size = label_id_size_pairs[label]     # given label size
    if not label_sets_dict.has_key(label_size):
      label_sets_dict[label_size] = set()
    label_sets_dict[label_size].add(doc_idx) # we aggregate document id

label_sizes = label_sets_dict.keys()
label_sizes.sort()

with open(path_to_output_lmrr,'w') as fout:
  lMRR = {}
  #for label_size, label_ids in label_sets_dict.items():
  for label_size in label_sizes:
    print 'Label size: %d ..' % label_size
    mrr = 0
    num_labels_computed = 0
    doc_indices = label_sets_dict[label_size]
    for doc_id in doc_indices:
      for label,rank in test_predictions[doc_id]:
        if label_id_size_pairs[int(label)] == label_size:
          #mrr += 1/float(int(rank)+1)
          mrr += (int(rank)+1)
          num_labels_computed += 1

    #lMRR[label_size] = mrr/float(len(doc_indices)*num_labels_computed)
    lMRR[label_size] = mrr/float(num_labels_computed)

    fout.write('%d: %f\n' % (label_size, lMRR[label_size]))
    fout.flush()
