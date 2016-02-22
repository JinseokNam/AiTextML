#include "hdf5corpus.hpp"
#include "utils.hpp"
#include "vocab.hpp"
#include "huff.hpp"

void HDF5Corpus::build_internals(bool in)
{
  // open HDF5 file
  br.reset(new BioASQHDF5Reader(m_corpus_filename));
  br->loadMetadata();

  // collect information which contains number of documents, words and labels
  m_word_vocab_size = br->getNumOfWords();
  m_label_vocab_size = br->getNumOfLabels();
  m_num_seen_labels = br->getNumOfSeenLabels();

  m_num_train_documents = br->getNumOfTrainInstances();
  m_num_valid_documents = br->getNumOfValidInstances();
  m_num_test_documents = br->getNumOfTestInstances();

  if(in)
  {
    br->buildMappingData();

    // build internal structures such as word vocabulary and label vocabulary
    // also need to create huffman tree based on word frequencies

    for (size_t i=0; i < (size_t) m_word_vocab_size; i++)
    {
      const std::string& word_name = br->getWordNameOfIndex(i);
      int word_freq = br->getWordFrequencyOfIndex(i);
      //LOG(INFO) << word_name << " : " << word_freq;

      vocab_ptr pt(new Vocab(word_name, word_freq));
      m_vocabulary.push_back(pt);

      //m_word2idx_map[word_name] = idx++;
      m_train_words += word_freq;
    }

    create_huffman_tree();
  }
}

void HDF5Corpus::create_huffman_tree()
{
  std::vector<int> frequencies;
  for (auto& x : m_vocabulary)
  {
    frequencies.push_back(x->m_freq);
  }
  LOG(INFO) << "Copied frequency distribution of words";

  HuffmanTree ht(frequencies);
  ht.build_huffman_tree();
  LOG(INFO) << "Created huffman tree";

  long long L = m_vocabulary.size();
  for(long long a = 0; a < L; a++) {
    m_vocabulary[a]->set_codeword(ht.getCodewordOfNodeAt(a));
    m_vocabulary[a]->set_inner_node_idx(ht.traverseInnerNodesOf(a));
  }
}

long long HDF5Corpus::get_vocabsize() const
{
  return (long long) m_word_vocab_size;
}

long long HDF5Corpus::get_num_labels() const
{
  return (long long) m_label_vocab_size;
}

long long HDF5Corpus::get_num_seen_labels() const
{
  return (long long) m_num_seen_labels;
}

long long HDF5Corpus::get_train_words() const
{
  return m_train_words;
}

long long HDF5Corpus::get_num_train_instances() const
{
  return (long long) m_num_train_documents;
}

long long HDF5Corpus::get_num_valid_instances() const
{
  return (long long) m_num_valid_documents;
}

long long HDF5Corpus::get_num_test_instances() const
{
  return (long long) m_num_test_documents;
}

std::vector<vocab_ptr>& HDF5Corpus::getVocabulary() 
{
  return m_vocabulary;
}

void HDF5Corpus::getTrainInstances(const long long base_idx, const int mb_sz, std::vector<std::vector<int> >& words_in_docs, std::vector<std::vector<int> >& labels_in_docs)
{
  br->getTrainInstances(base_idx, mb_sz, words_in_docs, labels_in_docs);
}

void HDF5Corpus::getValidInstances(const long long base_idx, const int mb_sz, std::vector<std::vector<int> >& words_in_docs, std::vector<std::vector<int> >& labels_in_docs)
{
  br->getValidInstances(base_idx, mb_sz, words_in_docs, labels_in_docs);
}

void HDF5Corpus::getTestInstances(const long long base_idx, const int mb_sz, std::vector<std::vector<int> >& words_in_docs, std::vector<std::vector<int> >& labels_in_docs)
{
  br->getTestInstances(base_idx, mb_sz, words_in_docs, labels_in_docs);
}

void HDF5Corpus::getTrainInstance(const long long doc_idx, std::vector<int>& words, std::vector<int>& labels)
{
  br->getTrainInstance(doc_idx, words, labels);
}

void HDF5Corpus::getValidInstance(const long long doc_idx, std::vector<int>& words, std::vector<int>& labels)
{
  br->getValidInstance(doc_idx, words, labels);
}

void HDF5Corpus::getTestInstance(const long long doc_idx, std::vector<int>& words, std::vector<int>& labels)
{
  br->getTestInstance(doc_idx, words, labels);
}

void HDF5Corpus::getWordSeqFromLabelDesc(const long long label_idx, std::vector<int>& word_seq)
{
  br->getWordSeqFromLabelDesc(label_idx, word_seq);
}

const std::string& HDF5Corpus::getDatasetFilename() const
{
  return m_corpus_filename;
}
