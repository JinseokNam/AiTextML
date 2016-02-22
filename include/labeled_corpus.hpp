#ifndef LABELED_CORPUS_HPP_
#define LABELED_CORPUS_HPP_

#include "corpus.hpp"

class LabeledCorpus : public Corpus
{
public:
  LabeledCorpus(std::string corpus_filename) : Corpus(corpus_filename), m_label_filename(""), m_label_desc_filename(""), m_label_map(), m_inv_label_map(), m_label_desc_map()  {}
  LabeledCorpus(std::string corpus_filename, std::string label_filename) : Corpus(corpus_filename), m_label_filename(label_filename), m_label_desc_filename(""), m_label_map(), m_inv_label_map(), m_label_desc_map() {}
  LabeledCorpus(std::string corpus_filename, std::string label_filename, std::string label_desc_filename) : Corpus(corpus_filename), m_label_filename(label_filename), m_label_desc_filename(label_desc_filename), m_label_map(), m_inv_label_map() {}
  void build_label_dictionary();
  std::string get_label_filename() const;
  std::string get_label_desc_filename() const;
  long long get_num_labels() const;

  std::map<std::string, long long>& getLabelDictionary();
  std::map<long long, std::string>& getInvLabelDictionary();
  std::map<std::string, std::string>& getLabelDescDictionary();

protected:
  std::string m_label_filename;
  std::string m_label_desc_filename;
  std::ifstream m_label_file;
  std::ifstream m_label_desc_file;

  std::map<std::string,long long> m_label_map;
  std::map<long long, std::string> m_inv_label_map;
  std::map<std::string,std::string> m_label_desc_map;
};

#endif
