#include "corpus.hpp"
#include "vocab.hpp"
#include "utils.hpp"
#include "huff.hpp"

Corpus::Corpus(std::string corpus_filename) : m_corpus_filename(corpus_filename), m_min_count(5),  m_vocabulary(0), m_num_lines(-1), m_train_words(0)
{
}

Corpus::~Corpus()
{
  if(m_corpus_file.is_open()) m_corpus_file.close();
}

std::string Corpus::get_corpus_filename() const
{
  return m_corpus_filename;
}

long long Corpus::get_vocabsize() const
{
  return (long long) m_vocabulary.size();
}

long long Corpus::getIndexOf(std::string &word)
{
  std::map<std::string,long long>::iterator it = m_word2idx_map.find(word);
  if (it != m_word2idx_map.end())
    return m_word2idx_map[word];
  else
    return getUNKIndex();
}

long long Corpus::getUNKIndex() const
{
  return (long long) m_word2idx_map.size()-1;
} 

std::vector<vocab_ptr>& Corpus::getVocabulary() 
{
  return m_vocabulary;
}

int Corpus::getCodeLen(long long word_idx) const
{
  return m_vocabulary[word_idx]->get_codelen();
}

long long Corpus::getInnerIndexOfCodeAt(long long word_idx, int k)
{
  return m_vocabulary[word_idx]->get_inner_node_idxAt(k);
}

void Corpus::build_vocab()
{
  build_vocab(m_min_count);
}

void Corpus::build_vocab(int min_count)
{
  m_min_count = min_count;    // ignore words occurring than the minimum count

  m_corpus_file.open(m_corpus_filename.c_str(), std::ifstream::in);
  if(!m_corpus_file.is_open()) 
  {
    std::stringstream ss;
    ss << "[" << __FILE__ << ":" << __LINE__ << "] File not found to open: " << m_corpus_filename;
    throw FileNotFoundException(ss.str());
  }

  LOG(INFO) << "Now building vocabulary from " << m_corpus_filename;

  std::map<std::string, int> temp_vocab;
  std::string line;
  long long num_lines_read = 0;
  while(std::getline(m_corpus_file, line))
  {
    num_lines_read++;
    std::vector<std::string> tokens = split(line.c_str(), ' ');
    for(std::vector<std::string>::iterator it = tokens.begin(); it != tokens.end(); ++it)
    {
      std::map<std::string, int>::iterator map_it = temp_vocab.find(*it);
      if(map_it == temp_vocab.end()) temp_vocab[*it] = 0;
      temp_vocab[*it] += 1;
    }
    if((num_lines_read % 1000000) == 0) LOG(INFO) << "Processed " << num_lines_read << " lines";
  }

  m_num_lines = num_lines_read;
  int unk_counts = 0;

  for (auto& x: temp_vocab)
  {
    if(x.second >= m_min_count)
    {
      vocab_ptr pt(new Vocab(x.first, x.second));
      m_vocabulary.push_back(pt);
    }
    else
    {
      unk_counts++;
    }
  }

  sortVocab();
  vocab_ptr pt(new Vocab(std::string("UNK"), unk_counts));
  m_vocabulary.push_back(pt);

  long long idx=0;

  for (auto& x : m_vocabulary)
  {
    m_word2idx_map[x->m_word] = idx++;
    m_train_words += x->m_freq;
  }

  create_huffman_tree();
}

void Corpus::sortVocab()
{
  VocabComp v_comp;
  std::sort(m_vocabulary.begin(), m_vocabulary.end(), 
      [&v_comp] (const vocab_ptr &l, const vocab_ptr &r)
      {
        return v_comp(*l.get(), *r.get());
      }
  );

  LOG(INFO) << "The vocabulary has been sorted by frequency.";
}

void Corpus::copyVocabularyFrom(Corpus& src)
{
  std::vector<vocab_ptr>& src_vocabulary = src.getVocabulary();

  long long idx=0;
  for (const auto& x : src_vocabulary)
  {
    vocab_ptr pt(new Vocab(x->m_word, x->m_freq));
    m_vocabulary.push_back(pt);

    m_word2idx_map[x->m_word] = idx++;
    m_train_words += x->m_freq;
  }

  LOG(INFO) << "Now counting the number of lines from " << m_corpus_filename;

  m_corpus_file.open(m_corpus_filename.c_str(), std::ifstream::in);
  if(!m_corpus_file.is_open()) 
  {
    std::stringstream ss;
    ss << "[" << __FILE__ << ":" << __LINE__ << "] File not found to open: " << m_corpus_filename;
    throw FileNotFoundException(ss.str());
  }

  std::string line;
  long long num_lines_read = 0;
  while(std::getline(m_corpus_file, line))
  {
    num_lines_read++;
  }
  m_corpus_file.close();

  m_num_lines = num_lines_read;

  create_huffman_tree();
}

void Corpus::recompute_frequency()
{
  LOG(INFO) << "Now re-computing frequency from " << m_corpus_filename;

  m_corpus_file.open(m_corpus_filename.c_str(), std::ifstream::in);
  if(!m_corpus_file.is_open()) 
  {
    std::stringstream ss;
    ss << "[" << __FILE__ << ":" << __LINE__ << "] File not found to open: " << m_corpus_filename;
    throw FileNotFoundException(ss.str());
  }

  for (auto& x : m_vocabulary)
  {
    x->m_freq = 0;
  }

  m_train_words = 0;
  std::string line;
  long long num_lines_read = 0;
  while(std::getline(m_corpus_file, line))
  {
    num_lines_read++;
    std::vector<std::string> tokens = split(line.c_str(), ' ');
    for(std::vector<std::string>::iterator it = tokens.begin(); it != tokens.end(); ++it)
    {
      std::map<std::string, long long>::iterator map_it = m_word2idx_map.find(*it);
      if(map_it == m_word2idx_map.end()) m_vocabulary[m_vocabulary.size()-1]->m_freq += 1;
      else m_vocabulary[map_it->second]->m_freq += 1;
      m_train_words++;
    }
    if((num_lines_read % 100000) == 0) LOG(INFO) << "Processed " << num_lines_read << " lines";
  }
}

void Corpus::save_vocab(std::string vocab_save_filepath)
{
  std::ofstream file;
  file.open(vocab_save_filepath);

  CHECK(file.is_open()) << "Failed to open the file to store vocabulary";

  for (auto& x : m_vocabulary)
  {
    file << x->m_word << "\t" << x->m_freq << std::endl;
  }

  file.close();
}

void Corpus::load_vocab(std::string vocab_load_filepath)
{
  std::ifstream file;
  file.open(vocab_load_filepath);

  CHECK(file.is_open()) << "Failed to open the file to load vocabulary";

  std::string line;
  std::vector<std::string> elems;
  while(std::getline(file, line))
  {
    elems.clear();
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, '\t'))
    {
      elems.push_back(item);
    }

    CHECK(elems.size() == 2) << "Error to parse: " << line;

    vocab_ptr pt(new Vocab(elems[0], std::stoi(elems[1]))); 
    m_vocabulary.push_back(pt); 
  }

  file.close();

  long long idx=0;

  for (auto& x : m_vocabulary)
  {
    m_word2idx_map[x->m_word] = idx++;
    m_train_words += x->m_freq;
  }

  // count the number of lines
  m_corpus_file.open(m_corpus_filename.c_str(), std::ifstream::in);
  if(!m_corpus_file.is_open()) 
  {
    std::stringstream ss;
    ss << "[" << __FILE__ << ":" << __LINE__ << "] File not found to open: " << m_corpus_filename;
    throw FileNotFoundException(ss.str());
  }

  LOG(INFO) << "Now counting the number of lines from " << m_corpus_filename;

  long long num_lines_read = 0;
  while(std::getline(m_corpus_file, line))
  {
    num_lines_read++;
  }
  m_corpus_file.close();

  m_num_lines = num_lines_read;

  create_huffman_tree();
}

void Corpus::create_huffman_tree()
{
  std::vector<int> frequencies;
  for (auto& x : m_vocabulary)
  {
    frequencies.push_back(x->m_freq);
  }
  LOG(INFO) << "Copied frequency distribution of words";

  HuffmanTree ht(frequencies);
  ht.build_huffman_tree();
  //ht.display_huffman_tree();
  LOG(INFO) << "Created huffman tree";

  long long L = m_vocabulary.size();
  for(long long a = 0; a < L; a++) {
    m_vocabulary[a]->set_codeword(ht.getCodewordOfNodeAt(a));
    m_vocabulary[a]->set_inner_node_idx(ht.traverseInnerNodesOf(a));
  }
}

long long Corpus::get_num_lines() const
{
  return m_num_lines;
}

long long Corpus::get_train_words() const
{
  return m_train_words;
}
