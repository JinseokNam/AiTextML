#ifndef HUFF_HPP_
#define HUFF_HPP_

#include "common.hpp"
#define MAX_CODEWORD 40
#define HUFF_LEFT 0
#define HUFF_RIGHT 1

typedef struct hnode_ {
  long long key;
  int freq;
  long long internal_idx;
  std::vector<char> codeword;
  boost::shared_ptr<struct hnode_> left;
  boost::shared_ptr<struct hnode_> right;
} Hnode;    // a node for huffman coding

typedef boost::shared_ptr<Hnode> hnode_ptr;

class CompareHnode {
public:
  // used in a priority queue (ascending order) 
  bool operator()(const Hnode& node1, const Hnode& node2) {
   if (node1.freq > node2.freq) return true;
   if (node1.freq < node2.freq) return false;
   return false;
  }
};

class HuffmanTree
{
public:
  HuffmanTree(std::vector<int> &frequencies);
  ~HuffmanTree();
  void build_huffman_tree();
  void display_huffman_tree();

  // accessor
  long long getKeyOfNodeAt(long long i); 
  int getFreqOfNodeAt(long long i);
  long long getInternalIndexOfNodeAt(long long i);
  const std::vector<char>& getCodewordOfNodeAt(long long i);

  std::vector<long long> traverseInnerNodesOf(long long i);

private:
  HuffmanTree(){}
  std::vector<hnode_ptr> m_hnodes;
  hnode_ptr m_tree_root;

  void generate_huffman_code(hnode_ptr root, std::stringstream &prefix);
  void traverse_huffman_tree(hnode_ptr root);
};

#endif
