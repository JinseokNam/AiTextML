#include "huff.hpp"

HuffmanTree::HuffmanTree(std::vector<int> &frequencies)
{
  unsigned int L = frequencies.size();
  for(unsigned int a = 0; a < L; a++)
  {
    hnode_ptr node(new Hnode());
    node->key = a;
    node->freq = frequencies[a];
    node->internal_idx = -1;
    node->left = NULL;
    node->right = NULL;
    
    m_hnodes.push_back(node);
  }
  CHECK_EQ(L,m_hnodes.size()) << "Both numbers should equal.";
}

HuffmanTree::~HuffmanTree()
{
}

void HuffmanTree::build_huffman_tree()
{
  long long idx = 0;

  CompareHnode h_comp;
  auto comp = [&h_comp] (const hnode_ptr &l, const hnode_ptr &r)
              {
                return h_comp(*l.get(), *r.get());
              };
  std::priority_queue<hnode_ptr, std::vector<hnode_ptr>, decltype( comp )> pq(comp);

  for(std::vector<hnode_ptr>::iterator it = m_hnodes.begin(); it != m_hnodes.end(); ++it)
  {
    pq.push(*it);
  } 

  hnode_ptr first,second;
  while(pq.size() > 1)
  {
    first = pq.top();
    pq.pop();
    second = pq.top();
    pq.pop();

    hnode_ptr new_merged(new Hnode());
    new_merged->key = -1;
    new_merged->internal_idx = idx++;
    new_merged->freq = first->freq + second->freq;
    new_merged->left = first;
    new_merged->right = second;

    pq.push(new_merged);
  }

  CHECK_EQ(pq.size(), 1u) << "Only one node in the priority queue";

  m_tree_root = pq.top();
  pq.pop();

  CHECK(pq.empty());

  std::stringstream prefix;
  generate_huffman_code(m_tree_root, prefix);
}

void HuffmanTree::generate_huffman_code(hnode_ptr root, std::stringstream &prefix)
{
  std::string prefix_str = prefix.str();
  std::copy(prefix_str.begin(), prefix_str.end(), back_inserter(root->codeword));;

  if(root->left == NULL || root->right == NULL) return;

  std::stringstream left_prefix;
  left_prefix << prefix.str() << HUFF_LEFT; 
  generate_huffman_code(root->left, left_prefix);

  std::stringstream right_prefix;
  right_prefix << prefix.str() << HUFF_RIGHT;
  generate_huffman_code(root->right, right_prefix);
}
 
void HuffmanTree::traverse_huffman_tree(hnode_ptr root)
{
  if(root->key != -1) {
    std::string s = std::string(root->codeword.begin(), root->codeword.end());
    std::cout << root->key << ":" << root->freq << " " << s << " " << s.size() << std::endl;
  } else {
    std::cout << "internal " << root->internal_idx << std::endl;
  }
  if(root->left == NULL || root->right == NULL) return;

  traverse_huffman_tree(root->left);
  traverse_huffman_tree(root->right);
}

void HuffmanTree::display_huffman_tree()
{
  LOG_IF(ERROR, !m_tree_root) << "Ignored a request to display an empty huffman tree";
  if(m_tree_root) traverse_huffman_tree(m_tree_root);
}

long long HuffmanTree::getKeyOfNodeAt(long long i)
{
  return m_hnodes[i]->key;
}

int HuffmanTree::getFreqOfNodeAt(long long i)
{
  return m_hnodes[i]->freq;
}

long long HuffmanTree::getInternalIndexOfNodeAt(long long i)
{
  return m_hnodes[i]->internal_idx;
}

const std::vector<char>& HuffmanTree::getCodewordOfNodeAt(long long i)
{
  return m_hnodes[i]->codeword;
}

std::vector<long long> HuffmanTree::traverseInnerNodesOf(long long i)
{
  hnode_ptr start = m_tree_root;
  std::vector<long long> node_indices;
  for(unsigned int b = 0; b < m_hnodes[i]->codeword.size(); b++)
  {
    CHECK_NE(start->internal_idx,-1);

    node_indices.push_back(start->internal_idx);
    if((m_hnodes[i]->codeword[b] - '0') == HUFF_LEFT) start = start->left;
    else if((m_hnodes[i]->codeword[b] - '0') == HUFF_RIGHT) start = start->right;
    else LOG(FATAL) << "expected: 0 or 1, actual: " << m_hnodes[i]->codeword[b];
  }
  return node_indices;
}
