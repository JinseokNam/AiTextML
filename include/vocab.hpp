#ifndef VOCAB_HPP_
#define VOCAB_HPP_

#include "common.hpp"

class Vocab
{
public:
  Vocab(std::string word, int freq);
  ~Vocab();
  std::string m_word;
  int m_freq;
  
  int get_codelen() const;

  void set_codeword(std::vector<char> codeword);
  int get_codeAt(int i);
  
  void set_inner_node_idx(std::vector<long long> inner_node_idx);
  long long get_inner_node_idxAt(int i);

  // testing purpose
  std::string get_codeword();
  std::string get_inner_node_idx();

private:
  Vocab();

  std::vector<char> m_codes;
  std::vector<long long> m_inner_node_idx;

  // Allow serialization to access non-public data members.  
  friend class boost::serialization::access; 

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & m_word;
    ar & m_freq;
    ar & m_codes;
    ar & m_inner_node_idx;
  }
};


class VocabComp
{
public:
  bool operator()(const Vocab& vocab1, const Vocab& vocab2) 
  {
    if(vocab1.m_freq < vocab2.m_freq) return false;
    if(vocab1.m_freq > vocab2.m_freq) return true;
    return false;
  }
};

#endif
