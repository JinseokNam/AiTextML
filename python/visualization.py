
# coding: utf-8
import numpy as np
from time import time
from sklearn.manifold import TSNE
from load_model import *

t0=time()
trained_model = load_model('D:\Bitbucket_VectorLearn\python\AAN_sentences_resulting_model.vectors')
wordamount=5000
tsne = TSNE(n_components=2)
mapped_data = tsne.fit_transform(trained_model['word_emb'][0:wordamount,:])
t1 = time()
print 'TSNE ',wordamount, ' takes ', (t1-t0)


get_ipython().magic(u'matplotlib inline')



import matplotlib.pyplot as plt

fig = plt.gcf()
fig.set_size_inches(18.5, 18.5)
plt.axis('off')
#plt.scatter(mapped_data[:,0], mapped_data[:,1])


###########

file=open(r'D:\Dropbox\01_KDSL\Input\aan_categories\topic.txt', 'r')
topic_keywords= file.readlines()
file=open(r'C:\Users\mazhe\OneDrive\collab JN_ZM\ParagraphVector_trained_model\AAN_word_vocab.txt','r')
aan_word_vocab= file.readlines()
for i in range(len(aan_word_vocab)):
        aan_word_vocab[i]=aan_word_vocab[i].split('\t')[0]
        

len1=len(mapped_data)
print 'mapped_data size:', len1
#iterate thru all words
for i in range(len1):
    x=mapped_data[i,0]
    y=mapped_data[i,1]
    word=aan_word_vocab[i]
    plt.text(x, y, word, color='blue')

    
fig.savefig(r'C:\Users\mazhe\OneDrive\collab JN_ZM\ParagraphVector_trained_model\word_plotting1.png')
plt.close(fig)
t2 = time()
print 'rest takes ', (t2-t1)
