#ifndef OHDATA_HPP_
#define OHDATA_HPP_

#include "hdata.hpp"

#include <string>
#include <vector>
#include <map>
#include <set>
#include <boost/shared_ptr.hpp>

#include "H5Cpp.h"

class BioASQHDF5Writer
{
public:
  BioASQHDF5Writer(const std::string traindata_path, const std::string trainlabel_path, const std::string validdata_path, const std::string validlabel_path, const std::string testdata_path, const std::string testlabel_path, const std::string label_descriptions, const std::string output_filepath);

  void build_word_vocabulary(const std::string word_vocabulary_filepath, const int mim_count);
  void build_label_vocabulary(const std::string label_vocabulary_filepath, const int mim_count);

  int convert2hdf5();

private:
  void build_vocabulary(const std::string vocabulary_filepath,
                        const int min_count,
                        std::vector<entity_ptr>& vocabulary,
                        std::map<std::string, long long>& entity2idx_map);

  int writeInstances(const std::string documents, const std::string labelset, hsize_t *set_base_idx, hsize_t *num_read_words_, hsize_t *num_read_labels_);
  int writeInstanceIndices(const std::string split_name, const std::string documents, hsize_t set_base_idx);
  int writeWordVocabulary();
  int writeLabelVocabulary(const std::string label_desc);

  void tokenization(const std::string input,
                    std::map<std::string, long long>& str2index_map,
                    std::vector<int>& token_indices);

  void incremental(H5::DataSpace& dataspace,
                   std::unique_ptr<H5::DataSet>& dataset,
                   const hsize_t *extended_data_size,            // size of the extended data space
                   const hsize_t *max_size,
                   const hsize_t *count,                         // number of elements to write 
                   const hsize_t *start,
                   const H5::DataType& mem_type,
                   const void *buf);

  OffsetInformation* createOffsetInformation(char* name, const unsigned long long offset, const int length);

  std::string m_traindata_path;
  std::string m_trainlabel_path;
  std::string m_validdata_path;
  std::string m_validlabel_path;
  std::string m_testdata_path;
  std::string m_testlabel_path;
  std::string m_label_descriptions_path;

  std::unique_ptr<H5::Group> m_split_group;
  std::unique_ptr<H5::DataSet> m_trainset_split_dataset;
  H5::DataSpace m_trainset_split_dataspace;
  std::unique_ptr<H5::DataSet> m_validset_split_dataset;
  H5::DataSpace m_validset_split_dataspace;
  std::unique_ptr<H5::DataSet> m_testset_split_dataset;
  H5::DataSpace m_testset_split_dataspace;

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

  int m_num_seen_labels;
  std::vector<entity_ptr> m_word_vocabulary;
  std::vector<entity_ptr> m_label_vocabulary;
  std::map<std::string,long long> m_word2idx_map;
  std::map<std::string,long long> m_label2idx_map;
  
  H5::H5File m_h5file;
  H5::CompType *offset_type, *freq_type;
};

#endif

