#!/etc/bin/python

import struct
import numpy as np

def load_model(model_path):
  with open(model_path,'rb') as f:
    is_train_labeled_data = struct.unpack('<1q',f.read(8))

    (dim, num_words) = struct.unpack('<2q',f.read(2*8))
    word_embeddings = np.fromfile(f, np.float32, count=dim*(num_words+2))
    word_embeddings = np.reshape(word_embeddings, (num_words+2, dim))
    
    (dim, num_documents) = struct.unpack('<2q',f.read(2*8))
    document_embeddings = np.fromfile(f, np.float32, count=dim*num_documents)
    document_embeddings = np.reshape(document_embeddings, (num_documents, dim))

    if is_train_labeled_data:
      (ldim, rdim) = struct.unpack('<2q',f.read(2*8))
      transform_matrix = np.fromfile(f, np.float32, count=ldim*rdim)
      transform_matrix = np.reshape(transform_matrix, (rdim, ldim))
      
      (dim, num_labels) = struct.unpack('<2q',f.read(2*8))
      label_embeddings = np.fromfile(f, np.float32, count=dim*num_labels)
      label_embeddings = np.reshape(label_embeddings, (num_labels, dim))
    
  word_emb_norms = np.apply_along_axis(np.linalg.norm, 1, word_embeddings)
  doc_emb_norms = np.apply_along_axis(np.linalg.norm, 1, document_embeddings)
  label_emb_norms = np.apply_along_axis(np.linalg.norm, 1, label_embeddings)

  model = dict()
  model['word_emb'] = word_embeddings/word_emb_norms.reshape(-1,1)
  model['doc_emb']= document_embeddings/doc_emb_norms.reshape(-1,1)
  if is_train_labeled_data:
    model['label_emb']= label_embeddings/label_emb_norms.reshape(-1,1)
    model['trans_mat']= transform_matrix

  return model

def load_inferred_test_instances(filepath):
  print 'Load the inferred vector representations for the test instances from %s' % filepath

  with open(filepath,'rb') as f:

    (dim, num_documents) = struct.unpack('<2q',f.read(2*8))
    document_embeddings = np.fromfile(f, np.float32, count=dim*num_documents)
    document_embeddings = np.reshape(document_embeddings, (num_documents, dim))
      
    (dim, num_labels) = struct.unpack('<2q',f.read(2*8))
    label_embeddings = np.fromfile(f, np.float32, count=dim*num_labels)
    label_embeddings = np.reshape(label_embeddings, (num_labels, dim))

  doc_emb_norms = np.apply_along_axis(np.linalg.norm, 1, document_embeddings)
  label_emb_norms = np.apply_along_axis(np.linalg.norm, 1, label_embeddings)

  test_set = dict()
  test_set['test_doc_emb']= document_embeddings/doc_emb_norms.reshape(-1,1)
  test_set['unseen_label_emb']= label_embeddings/label_emb_norms.reshape(-1,1)

  return test_set 
  

def display_similar_labels(label_embeddings, label_vocab, given_label_vec, k=10):
  assert(label_embeddings.shape[0] == len(label_vocab))

  sim_sorted_idx = np.argsort(np.sum(label_embeddings*given_label_vec,axis=1))[::-1][:k+1] 

  for i,ret_idx in enumerate(sim_sorted_idx):
    if i==0:
      continue
    print '%s:%f' % (label_vocab[ret_idx], np.sum(label_embeddings[ret_idx,:]*given_label_vec))

def eval_rankloss(sim_scores, targets):

  incorrect_pairwise_ranking = 0

  num_targets = len(targets)
  L = len(sim_scores)

  negative_scores = np.array(sim_scores)
  negative_scores[targets] = 0

  pos_label_ranks = []
  for pos_label in targets:
    pos_sim_score = sim_scores[pos_label]
    incorrect_pairs = np.sum(negative_scores>=pos_sim_score)
    pos_label_ranks.append((pos_label,incorrect_pairs))
    incorrect_pairwise_ranking += incorrect_pairs

  return incorrect_pairwise_ranking/float(num_targets*(L-num_targets)), pos_label_ranks

