#!/usr/bin/env python

#ranking_result = 'no_label_desc_ranking.txt'
#ranking_result = 'label_desc_ranking.txt'
#ranking_result = 'label_desc_ranking_latest.txt'
#ranking_result = 'label_desc_ranking_unseen_labels.txt'
ranking_result = 'label_desc_ranking_unseen_labels_by_words.txt'
with open(ranking_result) as fin:
  N = 0
  num_correct_predictions = 0
  for line in fin:
    if line.startswith('#'):
      continue

    N += 1
    for pair in line.strip().split(','):
      lid,rank = pair.split(':')
      if int(rank) == 0:
        num_correct_predictions += 1
        break

  print 1-num_correct_predictions/float(N)
        
