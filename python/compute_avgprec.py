#!/usr/bin/env python

#ranking_result = 'no_label_desc_ranking.txt'
#ranking_result = 'label_desc_ranking.txt'
#ranking_result = 'label_desc_ranking_latest.txt'
#ranking_result = 'label_desc_ranking_unseen_labels.txt'
ranking_result = 'label_desc_ranking_unseen_labels_by_words.txt'
with open(ranking_result) as fin:
  N = 0
  mean_avgprec = 0
  for line in fin:
    if line.startswith('#'):
      continue

    ranks = [int(pair.split(':')[1]) for pair in line.strip().split(',')]
    ranks.sort()
    avgprec = 0
    for i,rank in enumerate(ranks):
      avgprec += (i+1)/float(rank+1)  
    mean_avgprec += avgprec/float(len(ranks))
    N += 1

  print mean_avgprec/float(N)
        
