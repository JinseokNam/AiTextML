#ifndef IHDATA_HPP_
#define IHDATA_HPP_

#include "hdata.hpp"
#include <vector>
#include <map>
#include <set>
#include <boost/shared_ptr.hpp>

#include "H5Cpp.h"

class BioASQHDF5Reader
{
public:
  BioASQHDF5Reader(std::string hdf5_filepath);
  ~BioASQHDF5Reader();

  void loadMetadata();
  void buildMappingData();

  void getTrainInstances(const long long base_idx, const int mb_sz, std::vector<std::vector<int> >& words_in_docs, std::vector<std::vector<int> >& labels_in_docs);
  void getValidInstances(const long long base_idx, const int mb_sz, std::vector<std::vector<int> >& words_in_docs, std::vector<std::vector<int> >& labels_in_docs);
  void getTestInstances(const long long base_idx, const int mb_sz, std::vector<std::vector<int> >& words_in_docs, std::vector<std::vector<int> >& labels_in_docs);

  void getTrainInstance(const long long inst_idx, std::vector<int>& words, std::vector<int>& lables);
  void getValidInstance(const long long inst_idx, std::vector<int>& words, std::vector<int>& lables);
  void getTestInstance(const long long inst_idx, std::vector<int>& words, std::vector<int>& lables);

  void getWordSeqFromLabelDesc(const long long label_idx, std::vector<int>& word_seq);

  // Getter
  int getNumOfWords() const;
  int getNumOfLabels() const;
  int getNumOfSeenLabels() const;
  int getNumOfTrainInstances() const;
  int getNumOfValidInstances() const;
  int getNumOfTestInstances() const;

  // Vocabulary accessors
  const std::string& getWordNameOfIndex(int index);
  int getWordFrequencyOfIndex(int index);
  const std::string& getLabelNameOfIndex(int index);

private:
  void getWordSeqFromDocument(const long long doc_idx, std::vector<int>& word_seq);
  void getLabelsOfDocument(const long long doc_idx, std::vector<int>& labels);

  int m_num_words;
  int m_num_labels;
  int m_num_seen_labels;
  int m_num_train_instances;
  int m_num_valid_instances;
  int m_num_test_instances;
  int m_train_instance_offset;
  int m_valid_instance_offset;
  int m_test_instance_offset;

  H5::H5File m_h5file;

  std::unique_ptr<H5::Group> m_split_group;
  std::unique_ptr<H5::DataSet> m_trainset_split_dataset;
  H5::DataSpace m_trainset_split_dataspace;
  std::unique_ptr<H5::DataSet> m_validset_split_dataset;
  H5::DataSpace m_validset_split_dataspace;
  std::unique_ptr<H5::DataSet> m_testset_split_dataset;
  H5::DataSpace m_testset_split_dataspace;

  std::unique_ptr<H5::DataSpace> m_fixed_one_memspace;
  std::unique_ptr<H5::DataSpace> m_dyn_memspace;

  std::unique_ptr<H5::Group> m_words_group;
  std::unique_ptr<H5::DataSet> m_word_frequency_dataset;
  H5::DataSpace m_word_frequency_dataspace;

  std::unique_ptr<H5::Group> m_documents_group;
  std::unique_ptr<H5::DataSet> m_documents_offset_dataset;
  H5::DataSpace m_documents_offset_dataspace;
  std::unique_ptr<H5::DataSet> m_documents_wordseq_dataset;
  H5::DataSpace m_documents_wordseq_dataspace;

  std::unique_ptr<H5::Group> m_labels_group;
  std::unique_ptr<H5::DataSet> m_labels_offset_dataset;
  H5::DataSpace m_labels_offset_dataspace;
  std::unique_ptr<H5::DataSet> m_labels_wordseq_dataset;
  H5::DataSpace m_labels_wordseq_dataspace;

  std::unique_ptr<H5::Group> m_doc_label_pair_group;
  std::unique_ptr<H5::DataSet> m_doc_label_pair_offset_dataset;
  H5::DataSpace m_doc_label_pair_offset_dataspace;
  std::unique_ptr<H5::DataSet> m_doc_label_pair_lseq_dataset;
  H5::DataSpace m_doc_label_pair_lseq_dataspace;

  H5::CompType *offset_type, *freq_type;

  std::vector<std::pair<std::string, int> > m_word_name_freq_pairs;
  std::vector<std::string> m_label_names;
  std::vector<std::vector<int>*> m_label_descriptions;
  std::vector<std::pair<std::string, int> > m_train_instance_indices;
  std::vector<std::pair<std::string, int> > m_valid_instance_indices;
  std::vector<std::pair<std::string, int> > m_test_instance_indices;
};

#endif
