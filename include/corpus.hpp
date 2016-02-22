#ifndef CORPUS_HPP_
#define CORPUS_HPP_

#include "common.hpp"
#include "vocab.hpp"

typedef boost::shared_ptr<Vocab> vocab_ptr;

class Corpus
{
public:
  Corpus(std::string corpus_filename);
  ~Corpus();

  void build_vocab();
  void build_vocab(int min_count);
  void recompute_frequency();
  void save_vocab(std::string filepath);
  void load_vocab(std::string filepath);
  void create_huffman_tree(); 
  std::string get_corpus_filename() const;
  long long get_vocabsize() const;
  long long get_num_lines() const;
  long long getIndexOf(std::string &word);
  long long getUNKIndex() const;
  void copyVocabularyFrom(Corpus& src);

  std::vector<vocab_ptr>& getVocabulary();

  int getCodeLen(long long word_idx) const;
  long long getInnerIndexOfCodeAt(long long word_idx, int k);
  int get_codeAt(int i);

  long long get_train_words() const;


protected:
  //Corpus(const Corpus&) {}

  void sortVocab();

  std::string m_corpus_filename;
  std::ifstream m_corpus_file;
  int m_min_count;
  std::vector<vocab_ptr> m_vocabulary;
  long long m_num_lines;
  std::map<std::string,long long> m_word2idx_map;
  long long m_train_words;
};

#endif
