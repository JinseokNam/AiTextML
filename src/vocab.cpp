#include "vocab.hpp"

Vocab::Vocab()
      : m_word(""),
        m_freq(-1),
        m_codes(0),
        m_inner_node_idx(0)
{
}

Vocab::Vocab(std::string word, int freq) 
      : m_word(word),
        m_freq(freq),
        m_codes(0),
        m_inner_node_idx(0)
{
}

Vocab::~Vocab()
{
}

void Vocab::set_codeword(std::vector<char> codeword)
{
  m_codes.clear();
  std::copy(codeword.begin(), codeword.end(), back_inserter(m_codes));
}

std::string Vocab::get_codeword()
{
  std::string s(m_codes.begin(),m_codes.end());
  return s;
}

int Vocab::get_codeAt(int i)
{
  return m_codes[i] - '0';
}

int Vocab::get_codelen() const
{
  const int ret = m_codes.size();
  CHECK(ret > 0);
  return ret;
}

void Vocab::set_inner_node_idx(std::vector<long long> inner_node_idx)
{
  CHECK_EQ(inner_node_idx.size(), m_codes.size());
  m_inner_node_idx = inner_node_idx;
}

std::string Vocab::get_inner_node_idx()
{
  std::stringstream ss;
  std::copy(m_inner_node_idx.begin(), m_inner_node_idx.end(), std::ostream_iterator<long long>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);
  return s;
}

long long Vocab::get_inner_node_idxAt(int i)
{
  return m_inner_node_idx[i];
}
