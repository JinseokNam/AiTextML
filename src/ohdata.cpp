#include "ohdata.hpp"
#include "utils.hpp"
#include <glog/logging.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

BioASQHDF5Writer::BioASQHDF5Writer(
                    const std::string traindata_path, 
                    const std::string trainlabel_path, 
                    const std::string validdata_path, 
                    const std::string validlabel_path, 
                    const std::string testdata_path, 
                    const std::string testlabel_path, 
                    const std::string label_descriptions_path,
                    const std::string output_filepath)
                    : m_traindata_path(traindata_path),
                      m_trainlabel_path(trainlabel_path),
                      m_validdata_path(validdata_path),
                      m_validlabel_path(validlabel_path), m_testdata_path(testdata_path),
                      m_testlabel_path(testlabel_path),
                      m_label_descriptions_path(label_descriptions_path),
                      m_h5file(output_filepath, H5F_ACC_TRUNC),
                      offset_type(NULL)
                          
{
  hsize_t dims[1] = {0};
  hsize_t maxdims[1] = {H5S_UNLIMITED};
  hsize_t chunk_dims[1] = {500};

  std::unique_ptr<H5::DataSpace> dataspace(new H5::DataSpace ( 1, dims , maxdims));
  H5::DSetCreatPropList prop;
  prop.setChunk(1, chunk_dims);

  offset_type = new H5::CompType(sizeof(OffsetInformation));
  offset_type->insertMember(BioASQHDF5Data::member_name, HOFFSET(OffsetInformation, name), H5::StrType(H5::PredType::C_S1, H5T_VARIABLE));
  offset_type->insertMember(BioASQHDF5Data::member_offset, HOFFSET(OffsetInformation, offset), H5::PredType::NATIVE_ULLONG);
  offset_type->insertMember(BioASQHDF5Data::member_length, HOFFSET(OffsetInformation, length), H5::PredType::NATIVE_INT);

  freq_type = new H5::CompType(sizeof(FrequencyInformation));
  freq_type->insertMember(BioASQHDF5Data::member_name, HOFFSET(FrequencyInformation, name), H5::StrType(H5::PredType::C_S1, H5T_VARIABLE));
  freq_type->insertMember(BioASQHDF5Data::member_freq, HOFFSET(FrequencyInformation, freq), H5::PredType::NATIVE_INT);

  // train/test split
  m_split_group.reset(new H5::Group(m_h5file.createGroup(BioASQHDF5Data::DIR_SEP+BioASQHDF5Data::SPLIT_GROUP_NAME)));
  m_trainset_split_dataset.reset(new H5::DataSet(m_h5file.createDataSet(BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::SPLIT_GROUP_NAME
                                                                          +BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::TRAIN_SPLIT_DATASET_NAME
                                                                          , *offset_type, *dataspace, prop)));
  m_trainset_split_dataspace = m_trainset_split_dataset->getSpace();
  m_validset_split_dataset.reset(new H5::DataSet(m_h5file.createDataSet(BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::SPLIT_GROUP_NAME
                                                                          +BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::VALID_SPLIT_DATASET_NAME
                                                                          , *offset_type, *dataspace, prop)));
  m_validset_split_dataspace = m_validset_split_dataset->getSpace();
  m_testset_split_dataset.reset(new H5::DataSet(m_h5file.createDataSet(BioASQHDF5Data::DIR_SEP
                                                                            +BioASQHDF5Data::SPLIT_GROUP_NAME
                                                                            +BioASQHDF5Data::DIR_SEP
                                                                            +BioASQHDF5Data::TEST_SPLIT_DATASET_NAME
                                                                            , *offset_type, *dataspace, prop)));
  m_testset_split_dataspace = m_testset_split_dataset->getSpace();

  // word vocabulary consisting of frequency information
  m_words_group.reset(new H5::Group(m_h5file.createGroup(BioASQHDF5Data::DIR_SEP+BioASQHDF5Data::WORDS_GROUP_NAME)));
  m_word_frequency_dataset.reset(new H5::DataSet(m_h5file.createDataSet(BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::WORDS_GROUP_NAME
                                                                          +BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::FREQUENCY_DATASET_NAME
                                                                          , *freq_type, *dataspace, prop)));
  m_word_frequency_dataspace = m_word_frequency_dataset->getSpace();

  // document information
  m_documents_group.reset(new H5::Group(m_h5file.createGroup(BioASQHDF5Data::DIR_SEP+BioASQHDF5Data::DOCUMENTS_GROUP_NAME)));
  m_documents_offset_dataset.reset(new H5::DataSet(m_h5file.createDataSet(BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::DOCUMENTS_GROUP_NAME
                                                                          +BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::OFFSET_INFO_DATASET_NAME
                                                                          , *offset_type, *dataspace, prop)));
  m_documents_offset_dataspace = m_documents_offset_dataset->getSpace();
  m_documents_wordseq_dataset.reset(new H5::DataSet(m_h5file.createDataSet(BioASQHDF5Data::DIR_SEP
                                                                            +BioASQHDF5Data::DOCUMENTS_GROUP_NAME
                                                                            +BioASQHDF5Data::DIR_SEP
                                                                            +BioASQHDF5Data::ENTITY_INDICES_DATASET_NAME
                                                                            , H5::PredType::NATIVE_INT, *dataspace, prop)));
  m_documents_wordseq_dataspace = m_documents_wordseq_dataset->getSpace();

  m_labels_group.reset(new H5::Group(m_h5file.createGroup(BioASQHDF5Data::DIR_SEP+BioASQHDF5Data::LABELS_GROUP_NAME)));
  m_labels_offset_dataset.reset(new H5::DataSet(m_h5file.createDataSet(BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::LABELS_GROUP_NAME
                                                                          +BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::OFFSET_INFO_DATASET_NAME
                                                                          , *offset_type, *dataspace, prop)));
  m_labels_offset_dataspace = m_labels_offset_dataset->getSpace();
  m_labels_wordseq_dataset.reset(new H5::DataSet(m_h5file.createDataSet(BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::LABELS_GROUP_NAME
                                                                          +BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::ENTITY_INDICES_DATASET_NAME
                                                                          , H5::PredType::NATIVE_INT, *dataspace, prop)));
  m_labels_wordseq_dataspace = m_labels_wordseq_dataset->getSpace();

  m_doc_label_pair_group.reset(new H5::Group(m_h5file.createGroup(BioASQHDF5Data::DIR_SEP+BioASQHDF5Data::DOC_LABEL_PAIR_GROUP_NAME)));
  m_doc_label_pair_offset_dataset.reset(new H5::DataSet(m_h5file.createDataSet(BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::DOC_LABEL_PAIR_GROUP_NAME
                                                                          +BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::OFFSET_INFO_DATASET_NAME
                                                                          , *offset_type, *dataspace, prop)));
  m_doc_label_pair_offset_dataspace = m_doc_label_pair_offset_dataset->getSpace();
  m_doc_label_pair_lseq_dataset.reset(new H5::DataSet(m_h5file.createDataSet(BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::DOC_LABEL_PAIR_GROUP_NAME
                                                                          +BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::ENTITY_INDICES_DATASET_NAME
                                                                          , H5::PredType::NATIVE_INT, *dataspace, prop)));
  m_doc_label_pair_lseq_dataspace = m_doc_label_pair_lseq_dataset->getSpace();
}

void BioASQHDF5Writer::build_word_vocabulary(const std::string word_vocabulary_filepath, const int min_count)
{
  build_vocabulary(word_vocabulary_filepath, min_count, m_word_vocabulary, m_word2idx_map);
}

void BioASQHDF5Writer::build_label_vocabulary(const std::string label_vocabulary_filepath, const int min_count)
{
  static bool loaded_train_labels = false;
  if(m_label_vocabulary.size() > 0) loaded_train_labels = true;
  build_vocabulary(label_vocabulary_filepath, min_count, m_label_vocabulary, m_label2idx_map);
  if(!loaded_train_labels) m_num_seen_labels = m_label_vocabulary.size();
}

void BioASQHDF5Writer::build_vocabulary(const std::string vocabulary_filepath,
                                            const int min_count,
                                            std::vector<entity_ptr>& vocabulary,
                                            std::map<std::string, long long>& entity2idx_map)
{
  std::ifstream vocab_file(vocabulary_filepath, std::ifstream::in);
  CHECK(vocab_file.is_open()) << "File not found to open: " << vocabulary_filepath;

  LOG(INFO) << "Loading vocabulary from " << vocabulary_filepath;

  std::string line;
  std::vector<std::string> elems;
  long long idx=vocabulary.size(),unk_counts = 0;

  while(std::getline(vocab_file, line))
  {
    elems.clear();
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, '\t'))
    {
      elems.push_back(item);
    }

    CHECK(elems.size() == 2) << "Error to parse: " << line;
    std::string name(elems[0]);
    int freq = std::stoi(elems[1]);

    if(freq >= min_count && (entity2idx_map.find(name) == entity2idx_map.end()))
    {
      entity_ptr pt(new EntityInformation(name, freq, idx));
      vocabulary.push_back(pt);
      entity2idx_map[name] = idx;
      idx++;
    }
    else
    {
      unk_counts += freq;
    }
  }

  if(min_count > 1) 
  {
    entity_ptr pt(new EntityInformation(std::string("UNK"), unk_counts, idx));
    vocabulary.push_back(pt);
    entity2idx_map[std::string("UNK")] = idx;
  }

  vocab_file.close();
}

int BioASQHDF5Writer::convert2hdf5()
{
  hsize_t set_base_idx = 0;
  hsize_t num_read_words[1] = {0};
  hsize_t num_read_labels[1] = {0};

  // store train data
  writeInstanceIndices(std::string("trainset"), m_traindata_path, set_base_idx);
  writeInstances(m_traindata_path, m_trainlabel_path, &set_base_idx, num_read_words, num_read_labels);

  // store valid data
  if(exist_file(m_validdata_path) && exist_file(m_validlabel_path))
  {
    writeInstanceIndices(std::string("validset"), m_validdata_path, set_base_idx);
    writeInstances(m_validdata_path, m_validlabel_path, &set_base_idx, num_read_words, num_read_labels);
  }

  // store test data
  if(exist_file(m_testdata_path) && exist_file(m_testlabel_path))
  {
    writeInstanceIndices(std::string("testset"), m_testdata_path, set_base_idx);
    writeInstances(m_testdata_path, m_testlabel_path, &set_base_idx, num_read_words, num_read_labels);
  }

  writeWordVocabulary();
  writeLabelVocabulary(m_label_descriptions_path);

  return 0;
}

void BioASQHDF5Writer::incremental(H5::DataSpace& dataspace,
                                    std::unique_ptr<H5::DataSet>& dataset,
                                    const hsize_t *extended_data_size,     // size of the extended data space
                                    const hsize_t *max_size,
                                    const hsize_t *count,                  // number of elements to write 
                                    const hsize_t *start,
                                    const H5::DataType& mem_type,
                                    const void *buf
                                    )
{
  CHECK(*extended_data_size > 0 && *count > 0 && *start >= 0) << "extended data size: " << *extended_data_size << ", count: " << *count << ", start: " << *start;

  dataset->extend(extended_data_size); 
  dataspace.setExtentSimple(1, extended_data_size, max_size);

  // Select a hyperslab in extended portion of the word_seq_dataset.
  H5::DataSpace filespace(dataspace);
  filespace.selectHyperslab(H5S_SELECT_SET, count, start);

  // Define memory space.
  H5::DataSpace memspace(1, count, NULL);

  // Write data to the extended portion of the dataset.
  dataset->write(buf, mem_type, memspace, filespace);
}

void BioASQHDF5Writer::tokenization(const std::string input, std::map<std::string, long long>& str2index_map, std::vector<int>& token_indices)
{
  std::vector<std::string> tokens = split(input, ' ');
  for(std::vector<std::string>::iterator it = tokens.begin(); it != tokens.end(); ++it)
  {
    std::map<std::string, long long>::iterator map_it = str2index_map.find(*it);
    if(map_it == str2index_map.end()) token_indices.push_back(str2index_map.size() - 1);
    else token_indices.push_back(map_it->second);
  }
}

OffsetInformation* BioASQHDF5Writer::createOffsetInformation(char* name, const unsigned long long offset, const int length)
{
  OffsetInformation *offset_info = new OffsetInformation();
  
  offset_info->name = name;
  offset_info->length = length;
  offset_info->offset = offset;

  return offset_info;
}

int BioASQHDF5Writer::writeInstances(const std::string documents_path, const std::string labelset_path, hsize_t *set_base_index, hsize_t *num_read_words_, hsize_t *num_read_labels_)
{

  /////////////////////////////////
  //
  // preparation of indices
  //
  ////////////////////////////////
  const hsize_t maxdims[1] = {H5S_UNLIMITED};
  const hsize_t num_documents[1] = {1};
  const hsize_t num_labelsets[1] = {1};

  hsize_t *num_read_words = num_read_words_;
  hsize_t *num_read_labels = num_read_labels_;

  // text body
  hsize_t wordseq_offset[1] = {0};    // no!
  hsize_t num_words[1] = {0};         // no!

  // document offset information
  hsize_t document_offset[1] = {0};   // maybe?
  hsize_t num_read_docs[1] = {0};     // maybe?

  // label set
  hsize_t labelseq_offset[1] = {0};   // no!
  hsize_t num_labels[1] = {0};        // no!

  // labelset offset information
  hsize_t labelset_offset[1] = {0};   // maybe?
  hsize_t num_read_labelset[1] = {0}; // maybe?

  try
  {
    // Turn off the auto-printing when failure occurs so that we can
    // handle the errors appropriately
    H5::Exception::dontPrint();

    int document_base_idx = *set_base_index;
    int labelset_base_idx = *set_base_index;

    {
      std::ifstream documents_file(documents_path, std::ifstream::in);
      std::string line;
      const std::string delim(":::");
      std::vector<int> token_indices;
      while(std::getline(documents_file, line))
      {
        token_indices.clear();

        std::string doc_id = line.substr(0, line.find(delim));
        std::string text = line.substr(line.find(delim) + delim.length());

        tokenization(text, m_word2idx_map, token_indices);

        // Words in Documents
        wordseq_offset[0] = num_read_words[0];
        num_words[0] = token_indices.size();
        num_read_words[0] += token_indices.size();

        incremental(m_documents_wordseq_dataspace, m_documents_wordseq_dataset,
                    num_read_words, maxdims, num_words, wordseq_offset,
                    H5::PredType::NATIVE_INT, &token_indices[0]);

        std::string str_name = doc_id;
        std::unique_ptr<char[]> name(new char[str_name.length() + 1]);
        strcpy(name.get(), str_name.c_str());

        std::unique_ptr<OffsetInformation> doc_info(createOffsetInformation(name.get(), wordseq_offset[0], token_indices.size()));

        // Document Information
        document_offset[0] = document_base_idx;
        num_read_docs[0] = document_base_idx+1;

        incremental(m_documents_offset_dataspace, m_documents_offset_dataset,
                    num_read_docs, maxdims, num_documents, document_offset,
                    *offset_type, doc_info.get());

        document_base_idx++;
      }
    }

    /*
    *   Document-Labels Pair
    */
    {
      std::ifstream labelset_file(labelset_path, std::ifstream::in);
      std::string label_line;
      const std::string delim(":::");
      std::vector<int> label_indices;
      while(1)
      {
        if(!std::getline(labelset_file, label_line)) break;
        label_indices.clear();

        std::string doc_id = label_line.substr(0, label_line.find(delim));
        std::string text = label_line.substr(label_line.find(delim) + delim.length());

        tokenization(text, m_label2idx_map, label_indices);

        labelseq_offset[0] = num_read_labels[0];
        num_read_labels[0] += label_indices.size();
        num_labels[0] = label_indices.size();

        incremental(m_doc_label_pair_lseq_dataspace, m_doc_label_pair_lseq_dataset,
                    num_read_labels, maxdims, num_labels, labelseq_offset,
                    H5::PredType::NATIVE_INT, &label_indices[0]);

        std::string str_name = doc_id;
        std::unique_ptr<char[]> name(new char[str_name.length() + 1]);
        strcpy(name.get(), str_name.c_str());

        std::unique_ptr<OffsetInformation> doc_info(createOffsetInformation(name.get(), labelseq_offset[0], label_indices.size()));

        labelset_offset[0] = labelset_base_idx;
        num_read_labelset[0] = labelset_base_idx+1;

        incremental(m_doc_label_pair_offset_dataspace, m_doc_label_pair_offset_dataset,
                    num_read_labelset, maxdims, num_labelsets, labelset_offset,
                    *offset_type, doc_info.get());

        labelset_base_idx++;
      }
    }

    CHECK_EQ(document_base_idx, labelset_base_idx);
    *set_base_index = document_base_idx;    // index : this should be accumulated 

  }
  // catch failure caused by the H5File operations
  catch(H5::FileIException error)
  {
    error.printError();
    std::cerr << "FileExceptionError" << std::endl;
    return -1;
  }

  // catch failure caused by the DataSet operations
  catch(H5::DataSetIException error)
  {
    error.printError();
    std::cerr << "DatasetExceptionError" << std::endl;
    return -1;
  }

  return 0;
}

int BioASQHDF5Writer::writeInstanceIndices(std::string split_name, const std::string documents_path, hsize_t set_base_idx)
{
  int document_base_idx = 0;
  std::string trainset_split_name("trainset");
  std::string validset_split_name("validset");
  std::string testset_split_name("testset");

  const hsize_t maxdims[1] = {H5S_UNLIMITED};
  const hsize_t num_documents[1] = {1};

  // document offset information
  hsize_t document_offset[1] = {0};
  hsize_t num_read_docs[1] = {0};  

  try
  {
    // Turn off the auto-printing when failure occurs so that we can
    // handle the errors appropriately
    H5::Exception::dontPrint();
    {
      std::ifstream documents_file(documents_path, std::ifstream::in);
      std::string line;
      const std::string delim(":::");
      while(std::getline(documents_file, line))
      {
        std::string str_name = line.substr(0, line.find(delim));
        char *name = new char[str_name.length() + 1];
        strcpy(name, str_name.c_str());

        std::unique_ptr<OffsetInformation> doc_info(createOffsetInformation(name, set_base_idx+document_base_idx, 1));

        // Document Information
        document_offset[0] = document_base_idx;
        num_read_docs[0] = document_base_idx+1;

        if(split_name.compare(trainset_split_name) == 0)
        {
          incremental(m_trainset_split_dataspace, m_trainset_split_dataset,
                      num_read_docs, maxdims, num_documents, document_offset,
                      *offset_type, doc_info.get());
        }
        else if(split_name.compare(validset_split_name) == 0)
        { 
          incremental(m_validset_split_dataspace, m_validset_split_dataset,
                      num_read_docs, maxdims, num_documents, document_offset,
                      *offset_type, doc_info.get());
        }
        else if(split_name.compare(testset_split_name) == 0)
        { 
          incremental(m_testset_split_dataspace, m_testset_split_dataset,
                      num_read_docs, maxdims, num_documents, document_offset,
                      *offset_type, doc_info.get());
        }

        delete name;
        document_base_idx++;
      }

      int attr_data[1] = {0};
      attr_data[0] = document_base_idx;
      H5::DataSpace attr_dataspace = H5::DataSpace(H5S_SCALAR);

      if(split_name.compare(trainset_split_name) == 0)
      {
        H5::Attribute attribute = m_trainset_split_dataset->createAttribute(
                                    BioASQHDF5Data::NUM_INSTANCES_ATTR_NAME,
                                    H5::PredType::NATIVE_INT, 
                                    attr_dataspace);
        attribute.write( H5::PredType::NATIVE_INT, attr_data);
      }
      if(split_name.compare(validset_split_name) == 0)
      {
        H5::Attribute attribute = m_validset_split_dataset->createAttribute(
                                    BioASQHDF5Data::NUM_INSTANCES_ATTR_NAME,
                                    H5::PredType::NATIVE_INT, 
                                    attr_dataspace);
        attribute.write( H5::PredType::NATIVE_INT, attr_data);
      }
      else if(split_name.compare(testset_split_name) == 0)
      { 
        H5::Attribute attribute = m_testset_split_dataset->createAttribute(
                                    BioASQHDF5Data::NUM_INSTANCES_ATTR_NAME,
                                    H5::PredType::NATIVE_INT, 
                                    attr_dataspace);
        attribute.write( H5::PredType::NATIVE_INT, attr_data);
      }
    }
  }
  // catch failure caused by the H5File operations
  catch(H5::FileIException error)
  {
    error.printError();
    std::cerr << "FileExceptionError" << std::endl;
    return -1;
  }

  // catch failure caused by the DataSet operations
  catch(H5::DataSetIException error)
  {
    error.printError();
    std::cerr << "DatasetExceptionError" << std::endl;
    return -1;
  }

  return 0;
}

int BioASQHDF5Writer::writeWordVocabulary()
{
  const hsize_t maxdims[1] = {H5S_UNLIMITED};
  const hsize_t num_words[1] = {1};

  hsize_t word_offset[1] = {0};
  hsize_t num_read_words[1] = {0};

  try
  {
    // Turn off the auto-printing when failure occurs so that we can
    // handle the errors appropriately
    H5::Exception::dontPrint();

    for(size_t word_index = 0; word_index < m_word_vocabulary.size(); word_index++)
    {
      num_read_words[0] = word_index+1;
      word_offset[0] = word_index;

      std::string& word_name = m_word_vocabulary[word_index]->name;
      int& freq = m_word_vocabulary[word_index]->freq;

      std::unique_ptr<FrequencyInformation> word_info(new FrequencyInformation());
      char *name = new char[word_name.length() + 1];
      strcpy(name, word_name.c_str());
      word_info->name = name;
      word_info->freq = freq;

      incremental(m_word_frequency_dataspace, m_word_frequency_dataset,
                  num_read_words, maxdims, num_words, word_offset,
                  *freq_type, word_info.get());

      delete name;
    }
    {
      int attr_data[1] = {0};
      attr_data[0] = m_word_vocabulary.size();
      H5::DataSpace attr_dataspace = H5::DataSpace(H5S_SCALAR);

      H5::Attribute attribute = m_words_group->createAttribute(
                                  BioASQHDF5Data::NUM_WORDS_ATTR_NAME,
                                  H5::PredType::NATIVE_INT, 
                                  attr_dataspace);
      attribute.write( H5::PredType::NATIVE_INT, attr_data);
    }

  }
  // catch failure caused by the H5File operations
  catch(H5::FileIException error)
  {
    error.printError();
    std::cerr << "FileExceptionError" << std::endl;
    return -1;
  }

  // catch failure caused by the DataSet operations
  catch(H5::DataSetIException error)
  {
    error.printError();
    std::cerr << "DatasetExceptionError" << std::endl;
    return -1;
  }

  return 0;
}

int BioASQHDF5Writer::writeLabelVocabulary(const std::string label_descriptions_path)
{
  const hsize_t maxdims[1] = {H5S_UNLIMITED};
  const hsize_t num_labels[1] = {1};

  hsize_t wordseq_offset[1] = {0};
  hsize_t num_read_words[1] = {0};
  hsize_t num_words[1] = {0};

  hsize_t label_offset[1] = {0};
  hsize_t num_read_labels[1] = {0};

  std::ifstream label_desc_file(label_descriptions_path, std::ifstream::in);
  std::string line;
  const std::string delim(":::");
  std::map<std::string, std::string> label_desc_map;
  while(std::getline(label_desc_file, line))
  {
    std::string label_name = line.substr(0, line.find(delim));
    std::string description = line.substr(line.find(delim) + delim.length());

    if(m_label2idx_map.find(label_name) != m_label2idx_map.end())   // exist a label in the vocabulary
    {
      label_desc_map[label_name] = description;   // add label's description to the map
    }
  }

  try
  {
    // Turn off the auto-printing when failure occurs so that we can
    // handle the errors appropriately
    H5::Exception::dontPrint();

    std::vector<int> token_indices;
    for(size_t label_index = 0; label_index < m_label_vocabulary.size(); label_index++)
    {
      token_indices.clear();

      std::string label_name = m_label_vocabulary[label_index]->name;
      std::map< std::string, std::string >::iterator it = label_desc_map.find(label_name);
      std::string description = it->second;

      tokenization(description, m_word2idx_map, token_indices);

      wordseq_offset[0] = num_read_words[0];
      num_read_words[0] += token_indices.size();
      num_words[0] = token_indices.size();

      incremental(m_labels_wordseq_dataspace, m_labels_wordseq_dataset,
                  num_read_words, maxdims, num_words, wordseq_offset,
                  H5::PredType::NATIVE_INT, &token_indices[0]);

      num_read_labels[0] = label_index+1;
      label_offset[0] = label_index;

      std::unique_ptr<OffsetInformation> label_info(new OffsetInformation());
      char *name = new char[label_name.length() + 1];
      strcpy(name, label_name.c_str());
      label_info->name = name;
      label_info->length = token_indices.size();
      label_info->offset = wordseq_offset[0];

      incremental(m_labels_offset_dataspace, m_labels_offset_dataset,
                  num_read_labels, maxdims, num_labels, label_offset,
                  *offset_type, label_info.get());

      delete name;
    }

    // write atttributes
    {
      int attr_data[1] = {0};
      H5::DataSpace attr_dataspace = H5::DataSpace(H5S_SCALAR);

      attr_data[0] = m_label_vocabulary.size();
      H5::Attribute total_labels_attribute = m_labels_group->createAttribute(
                                      BioASQHDF5Data::NUM_LABELS_ATTR_NAME,
                                      H5::PredType::NATIVE_INT, 
                                      attr_dataspace);
      total_labels_attribute.write( H5::PredType::NATIVE_INT, attr_data);

      attr_data[0] = m_num_seen_labels;
      H5::Attribute seen_labels_attribute = m_labels_group->createAttribute(
                                      BioASQHDF5Data::NUM_SEEN_LABELS_ATTR_NAME,
                                      H5::PredType::NATIVE_INT, 
                                      attr_dataspace);
      seen_labels_attribute.write( H5::PredType::NATIVE_INT, attr_data);
    }

  }
  // catch failure caused by the H5File operations
  catch(H5::FileIException error)
  {
    error.printError();
    std::cerr << "FileExceptionError" << std::endl;
    return -1;
  }

  // catch failure caused by the DataSet operations
  catch(H5::DataSetIException error)
  {
    error.printError();
    std::cerr << "DatasetExceptionError" << std::endl;
    return -1;
  }

  return 0;
}

