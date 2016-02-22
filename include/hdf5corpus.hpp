#ifndef HDF5CORPUS_HPP_
#define HDF5CORPUS_HPP_

#include "common.hpp"
#include "vocab.hpp"
#include "ihdata.hpp"

typedef boost::shared_ptr<Vocab> vocab_ptr;

class HDF5Corpus
{
public:
  HDF5Corpus(std::string corpus_filename) : m_corpus_filename(corpus_filename), br(nullptr), m_word_vocab_size(0), m_label_vocab_size(0), m_num_seen_labels(0), m_num_train_documents(0), m_num_valid_documents(0), m_train_words(0), m_vocabulary(0), m_word2idx_map(), m_label_map(), m_inv_label_map(), m_label_desc_map()  {}
  
  void build_internals(bool in);
  long long get_train_words() const;
  long long get_num_labels() const;
  long long get_num_seen_labels() const;
  long long get_vocabsize() const;
  long long get_num_train_instances() const;
  long long get_num_valid_instances() const;
  long long get_num_test_instances() const;

  std::vector<vocab_ptr>& getVocabulary();

  void getTrainInstances(const long long base_idx, const int mb_sz, std::vector<std::vector<int> >& words_in_docs, std::vector<std::vector<int> >& labels_in_docs);
  void getValidInstances(const long long base_idx, const int mb_sz, std::vector<std::vector<int> >& words_in_docs, std::vector<std::vector<int> >& labels_in_docs);
  void getTestInstances(const long long base_idx, const int mb_sz, std::vector<std::vector<int> >& words_in_docs, std::vector<std::vector<int> >& labels_in_docs);

  void getTrainInstance(const long long doc_idx, std::vector<int>& words, std::vector<int>& labels);
  void getValidInstance(const long long doc_idx, std::vector<int>& words, std::vector<int>& labels);
  void getTestInstance(const long long doc_idx, std::vector<int>& words, std::vector<int>& labels);

  void getWordSeqFromLabelDesc(const long long label_idx, std::vector<int>& word_seq);

  const std::string& getDatasetFilename() const;

protected:
  std::string m_corpus_filename;
  std::unique_ptr<BioASQHDF5Reader> br;
  long long m_word_vocab_size;
  long long m_label_vocab_size;
  long long m_num_seen_labels;
  long long m_num_train_documents;
  long long m_num_valid_documents;
  long long m_num_test_documents;
  long long m_train_words;

  std::vector<vocab_ptr> m_vocabulary;
  std::map<std::string, long long> m_word2idx_map;
  std::map<std::string, long long> m_label_map;
  std::map<long long, std::string> m_inv_label_map;
  std::map<std::string,std::string> m_label_desc_map;

  void create_huffman_tree();
};

#endif
