#include "ihdata.hpp"
#include "utils.hpp"
#include <glog/logging.h>
#include <fstream>
#include <sstream>
#include <stdexcept>

BioASQHDF5Reader::BioASQHDF5Reader(std::string hdf5_filepath) : 
                                              m_num_words(0),
                                              m_num_labels(0),
                                              m_num_seen_labels(0),
                                              m_num_train_instances(0),
                                              m_num_valid_instances(0),
                                              m_num_test_instances(0),
                                              m_train_instance_offset(0),
                                              m_valid_instance_offset(0),
                                              m_test_instance_offset(0),
                                              m_h5file(hdf5_filepath, H5F_ACC_RDONLY),
                                              m_word_name_freq_pairs(0),
                                              m_label_names(0),
                                              m_label_descriptions(0)
{
  const hsize_t one_item[1] = {1};
  hsize_t dims[1] = {0};
  hsize_t maxdims[1] = {H5S_UNLIMITED};

  std::unique_ptr<H5::DataSpace> dataspace(new H5::DataSpace ( 1, dims , maxdims));

  offset_type = new H5::CompType(sizeof(OffsetInformation));
  offset_type->insertMember(BioASQHDF5Data::member_name, HOFFSET(OffsetInformation, name), H5::StrType(H5::PredType::C_S1, H5T_VARIABLE));
  offset_type->insertMember(BioASQHDF5Data::member_offset, HOFFSET(OffsetInformation, offset), H5::PredType::NATIVE_ULLONG);
  offset_type->insertMember(BioASQHDF5Data::member_length, HOFFSET(OffsetInformation, length), H5::PredType::NATIVE_INT);

  freq_type = new H5::CompType(sizeof(FrequencyInformation));
  freq_type->insertMember(BioASQHDF5Data::member_name, HOFFSET(FrequencyInformation, name), H5::StrType(H5::PredType::C_S1, H5T_VARIABLE));
  freq_type->insertMember(BioASQHDF5Data::member_freq, HOFFSET(FrequencyInformation, freq), H5::PredType::NATIVE_INT);

  m_fixed_one_memspace.reset(new H5::DataSpace(1, one_item, NULL));  // memory space
  m_dyn_memspace.reset(new H5::DataSpace(1, one_item, NULL));  // memory space

  m_split_group.reset(new H5::Group(m_h5file.openGroup(BioASQHDF5Data::DIR_SEP+BioASQHDF5Data::SPLIT_GROUP_NAME)));
  m_trainset_split_dataset.reset(new H5::DataSet(m_h5file.openDataSet(BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::SPLIT_GROUP_NAME
                                                                          +BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::TRAIN_SPLIT_DATASET_NAME)));
                                                                     
  m_trainset_split_dataspace = m_trainset_split_dataset->getSpace();
  m_validset_split_dataset.reset(new H5::DataSet(m_h5file.openDataSet(BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::SPLIT_GROUP_NAME
                                                                          +BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::VALID_SPLIT_DATASET_NAME)));
  m_validset_split_dataspace = m_validset_split_dataset->getSpace();
  m_testset_split_dataset.reset(new H5::DataSet(m_h5file.openDataSet(BioASQHDF5Data::DIR_SEP
                                                                            +BioASQHDF5Data::SPLIT_GROUP_NAME
                                                                            +BioASQHDF5Data::DIR_SEP
                                                                            +BioASQHDF5Data::TEST_SPLIT_DATASET_NAME)));
  m_testset_split_dataspace = m_testset_split_dataset->getSpace();

  // word vocabulary consisting of frequency information
  m_words_group.reset(new H5::Group(m_h5file.openGroup(BioASQHDF5Data::DIR_SEP+BioASQHDF5Data::WORDS_GROUP_NAME)));
  m_word_frequency_dataset.reset(new H5::DataSet(m_h5file.openDataSet(BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::WORDS_GROUP_NAME
                                                                          +BioASQHDF5Data::DIR_SEP
                                                                          +BioASQHDF5Data::FREQUENCY_DATASET_NAME)));
  m_word_frequency_dataspace = m_word_frequency_dataset->getSpace();

  m_documents_group.reset(new H5::Group(m_h5file.openGroup(BioASQHDF5Data::DIR_SEP+BioASQHDF5Data::DOCUMENTS_GROUP_NAME)));
  m_documents_offset_dataset.reset(new H5::DataSet(m_documents_group->openDataSet(BioASQHDF5Data::OFFSET_INFO_DATASET_NAME)));
  m_documents_offset_dataspace = m_documents_offset_dataset->getSpace();
  m_documents_wordseq_dataset.reset(new H5::DataSet(m_documents_group->openDataSet(BioASQHDF5Data::ENTITY_INDICES_DATASET_NAME)));
  m_documents_wordseq_dataspace = m_documents_wordseq_dataset->getSpace();

  m_labels_group.reset(new H5::Group(m_h5file.openGroup(BioASQHDF5Data::DIR_SEP+BioASQHDF5Data::LABELS_GROUP_NAME)));
  m_labels_offset_dataset.reset(new H5::DataSet(m_labels_group->openDataSet(BioASQHDF5Data::OFFSET_INFO_DATASET_NAME)));
  m_labels_offset_dataspace = m_labels_offset_dataset->getSpace();
  m_labels_wordseq_dataset.reset(new H5::DataSet(m_labels_group->openDataSet(BioASQHDF5Data::ENTITY_INDICES_DATASET_NAME)));
  m_labels_wordseq_dataspace = m_labels_wordseq_dataset->getSpace();

  m_doc_label_pair_group.reset(new H5::Group(m_h5file.openGroup(BioASQHDF5Data::DIR_SEP+BioASQHDF5Data::DOC_LABEL_PAIR_GROUP_NAME)));
  m_doc_label_pair_offset_dataset.reset(new H5::DataSet(m_doc_label_pair_group->openDataSet(BioASQHDF5Data::OFFSET_INFO_DATASET_NAME)));
  m_doc_label_pair_offset_dataspace = m_doc_label_pair_offset_dataset->getSpace();
  m_doc_label_pair_lseq_dataset.reset(new H5::DataSet(m_doc_label_pair_group->openDataSet(BioASQHDF5Data::ENTITY_INDICES_DATASET_NAME)));
  m_doc_label_pair_lseq_dataspace = m_doc_label_pair_lseq_dataset->getSpace();
}

BioASQHDF5Reader::~BioASQHDF5Reader()
{
  m_split_group->close();
  m_trainset_split_dataset->close();
  m_validset_split_dataset->close();
  m_testset_split_dataset->close();

  m_words_group->close();
  m_word_frequency_dataset->close();

  m_documents_offset_dataset->close();
  m_documents_wordseq_dataset->close();
  m_documents_group->close();

  m_labels_offset_dataset->close();
  m_labels_wordseq_dataset->close();
  m_labels_group->close();

  m_doc_label_pair_offset_dataset->close();
  m_doc_label_pair_lseq_dataset->close();
  m_doc_label_pair_group->close();

  m_h5file.close();
}

void BioASQHDF5Reader::loadMetadata()
{
  //DLOG(INFO) << "Started to load meta data from " << m_h5file.getFileName();

  H5::Attribute attr_num_words = m_words_group->openAttribute(BioASQHDF5Data::NUM_WORDS_ATTR_NAME); 
  attr_num_words.read( H5::PredType::NATIVE_INT, &m_num_words);
  
  H5::Attribute attr_num_labels = m_labels_group->openAttribute(BioASQHDF5Data::NUM_LABELS_ATTR_NAME); 
  attr_num_labels.read( H5::PredType::NATIVE_INT, &m_num_labels);

  H5::Attribute attr_num_seen_labels = m_labels_group->openAttribute(BioASQHDF5Data::NUM_SEEN_LABELS_ATTR_NAME); 
  attr_num_seen_labels.read( H5::PredType::NATIVE_INT, &m_num_seen_labels);

  H5::Attribute attr_num_train_insts = m_trainset_split_dataset->openAttribute(BioASQHDF5Data::NUM_INSTANCES_ATTR_NAME); 
  attr_num_train_insts.read( H5::PredType::NATIVE_INT, &m_num_train_instances);

  H5::Attribute attr_num_valid_insts = m_validset_split_dataset->openAttribute(BioASQHDF5Data::NUM_INSTANCES_ATTR_NAME); 
  attr_num_valid_insts.read( H5::PredType::NATIVE_INT, &m_num_valid_instances);

  H5::Attribute attr_num_test_insts = m_testset_split_dataset->openAttribute(BioASQHDF5Data::NUM_INSTANCES_ATTR_NAME); 
  attr_num_test_insts.read( H5::PredType::NATIVE_INT, &m_num_test_instances);

  m_train_instance_offset = 0;
  m_valid_instance_offset = m_train_instance_offset + m_num_train_instances;
  m_test_instance_offset = m_valid_instance_offset + m_num_valid_instances;

  //DLOG(INFO) << "Loaded meta data";
}

void BioASQHDF5Reader::buildMappingData()
{
  {

    const hsize_t num_word_vocab_elements = (hsize_t) m_num_words;
    const hsize_t num_label_elements = (hsize_t) m_num_labels;
    const hsize_t num_train_elements = (hsize_t) m_num_train_instances;
    const hsize_t num_valid_elements = (hsize_t) m_num_valid_instances;
    const hsize_t num_test_elements = (hsize_t) m_num_test_instances;
    const hsize_t base = 0;

    H5::DataSpace word_vocab_memspace(1, &num_word_vocab_elements, NULL);
    std::unique_ptr<FrequencyInformation[]> word_freq_info(new FrequencyInformation[num_word_vocab_elements]);
    m_word_frequency_dataspace.selectHyperslab(H5S_SELECT_SET, &num_word_vocab_elements, &base);
    m_word_frequency_dataset->read(word_freq_info.get(), *freq_type, word_vocab_memspace, m_word_frequency_dataspace);

    for(hsize_t i=0; i < num_word_vocab_elements; i++)
    {
      //m_word_frequency_dataspace.selectHyperslab(H5S_SELECT_SET, &num_elements, &i);
      //m_word_frequency_dataset->read(&word_freq_info, *freq_type, freq_memspace, m_word_frequency_dataspace);
      //LOG(INFO) << word_freq_info.name << " : " << word_freq_info.freq;
      m_word_name_freq_pairs.push_back(std::make_pair(word_freq_info.get()[i].name, word_freq_info.get()[i].freq));  
    }
    H5::DataSet::vlenReclaim(word_freq_info.get(), *freq_type, word_vocab_memspace);
    word_vocab_memspace.close();
    DLOG(INFO) << "Loaded word vocabulary";
    
    H5::DataSpace label_memspace(1, &num_label_elements, NULL);
    std::unique_ptr<OffsetInformation[]> label_offset_info(new OffsetInformation[num_label_elements]);
    m_labels_offset_dataspace.selectHyperslab(H5S_SELECT_SET, &num_label_elements, &base);
    m_labels_offset_dataset->read(label_offset_info.get(), *offset_type, label_memspace, m_labels_offset_dataspace);

    for(hsize_t i=0; i < (hsize_t) m_num_labels; i++)
    {
      m_label_names.push_back(label_offset_info.get()[i].name);
    }
    H5::DataSet::vlenReclaim(label_offset_info.get(), *offset_type, label_memspace);
    label_memspace.close();
    DLOG(INFO) << "Loaded label vocabulary";

    // TODO make the following code concise
    // We don't need to keep all indices on memory
    // in this code, just check whether indices are contigous or not

    H5::DataSpace train_memspace(1, &num_train_elements, NULL);
    std::unique_ptr<OffsetInformation[]> trainsplit_info(new OffsetInformation[m_num_train_instances]);
    m_trainset_split_dataspace.selectHyperslab(H5S_SELECT_SET, &num_train_elements, &base);
    m_trainset_split_dataset->read(trainsplit_info.get(), *offset_type, train_memspace, m_trainset_split_dataspace);

    for(hsize_t i=0; i < (hsize_t) m_num_train_instances; i++)
    {
      m_train_instance_indices.push_back(std::make_pair(trainsplit_info.get()[i].name, trainsplit_info.get()[i].offset));  
      //free(trainsplit_info.get()[i].name);
      //LOG(INFO) << trainsplit_info.get()[i].name << " : " << trainsplit_info.get()[i].offset;
      //LOG(INFO) << m_train_instance_indices[i].first << " : " << m_train_instance_indices[i].second;
    }
    H5::DataSet::vlenReclaim(trainsplit_info.get(), *offset_type, train_memspace);
    train_memspace.close();

    H5::DataSpace valid_memspace(1, &num_valid_elements, NULL);
    std::unique_ptr<OffsetInformation[]> validsplit_info(new OffsetInformation[m_num_valid_instances]);
    m_validset_split_dataspace.selectHyperslab(H5S_SELECT_SET, &num_valid_elements, &base);
    m_validset_split_dataset->read(validsplit_info.get(), *offset_type, valid_memspace, m_validset_split_dataspace);

    for(hsize_t i=0; i < (hsize_t) m_num_valid_instances; i++)
    {
      m_valid_instance_indices.push_back(std::make_pair(validsplit_info.get()[i].name, validsplit_info.get()[i].offset));  
    }
    H5::DataSet::vlenReclaim(validsplit_info.get(), *offset_type, valid_memspace);
    valid_memspace.close();

    H5::DataSpace test_memspace(1, &num_test_elements, NULL);
    std::unique_ptr<OffsetInformation[]> testsplit_info(new OffsetInformation[m_num_test_instances]);
    m_testset_split_dataspace.selectHyperslab(H5S_SELECT_SET, &num_test_elements, &base);
    m_testset_split_dataset->read(testsplit_info.get(), *offset_type, test_memspace, m_testset_split_dataspace);

    for(hsize_t i=0; i < (hsize_t) m_num_test_instances; i++)
    {
      m_test_instance_indices.push_back(std::make_pair(testsplit_info.get()[i].name, testsplit_info.get()[i].offset));  
    }
    H5::DataSet::vlenReclaim(testsplit_info.get(), *offset_type, test_memspace);
    test_memspace.close();
  }

  {
    hsize_t label_offset(0);
    hsize_t num_elements(m_num_labels);

    H5::DataSpace labels_memspace(1, &num_elements, NULL);
    std::unique_ptr<OffsetInformation[]> labels_info(new OffsetInformation[num_elements]);

    // Select a hyperslab in extended portion of the dataset.
    m_labels_offset_dataspace.selectHyperslab(H5S_SELECT_SET, &num_elements, &label_offset);
    m_labels_offset_dataset->read(labels_info.get(), *offset_type, labels_memspace, m_labels_offset_dataspace);

    //  Look up actual word indices
    hsize_t start_label_offset = labels_info.get()[0].offset;
    hsize_t total_length = labels_info.get()[num_elements-1].offset - labels_info.get()[0].offset + labels_info.get()[num_elements-1].length;

    CHECK(start_label_offset >= 0 && total_length >= 0);

    m_dyn_memspace->setExtentSimple(1, &total_length, NULL);
    m_labels_wordseq_dataspace.selectHyperslab(H5S_SELECT_SET, &total_length, &start_label_offset);

    std::unique_ptr<int[]> wordseq(new int[total_length]);    // dynamic memory space for a word sequence
    m_labels_wordseq_dataset->read(wordseq.get(), H5::PredType::NATIVE_INT, *m_dyn_memspace, m_labels_wordseq_dataspace);

    // put words into the output container
    size_t s_idx = 0, e_idx=0;
    
    for(size_t i=0; i < num_elements; i++)
    {
      e_idx = s_idx + labels_info[i].length;
      //LOG(INFO) << i+1 << "/" << num_elements << " >> " << s_idx << ":" << e_idx;
      m_label_descriptions.push_back(new std::vector<int>(wordseq.get()+s_idx, wordseq.get()+e_idx));
      s_idx = e_idx;
    }

    H5::DataSet::vlenReclaim(labels_info.get(), *offset_type, labels_memspace);
  }
}

void BioASQHDF5Reader::getWordSeqFromDocument(const long long doc_idx, std::vector<int>& wordseq_vec)
{
  if(wordseq_vec.size() > 0)
    wordseq_vec.clear();

  hsize_t doc_offset(doc_idx);
  hsize_t num_elements(1);

  OffsetInformation doc_info;
  // Select a hyperslab in extended portion of the dataset.
  m_documents_offset_dataspace.selectHyperslab(H5S_SELECT_SET, &num_elements, &doc_offset);
  m_documents_offset_dataset->read(&doc_info, *offset_type, *m_fixed_one_memspace, m_documents_offset_dataspace);

  //  Look up actual word indices
  hsize_t offset(doc_info.offset);
  hsize_t length(doc_info.length);

  CHECK(doc_info.offset >= 0 && doc_info.length >= 0);

  m_dyn_memspace->setExtentSimple(1, &length, NULL);
  m_documents_wordseq_dataspace.selectHyperslab(H5S_SELECT_SET, &length, &offset);

  std::unique_ptr<int[]> wordseq(new int[doc_info.length]);    // dynamic memory space for a word sequence
  m_documents_wordseq_dataset->read(wordseq.get(), H5::PredType::NATIVE_INT, *m_dyn_memspace, m_documents_wordseq_dataspace);
  
  // put words into the output container
  wordseq_vec.assign(wordseq.get(), wordseq.get() + doc_info.length);
}

void BioASQHDF5Reader::getWordSeqFromLabelDesc(const long long label_idx, std::vector<int>& word_seq)
{
  if(word_seq.size() > 0)
    word_seq.clear();
 
/*
  hsize_t label_offset(label_idx);
  hsize_t num_elements(1);

  OffsetInformation label_info;
  // Select a hyperslab in extended portion of the dataset.
  m_labels_offset_dataspace.selectHyperslab(H5S_SELECT_SET, &num_elements, &label_offset);
  m_labels_offset_dataset->read(&label_info, *offset_type, *m_fixed_one_memspace, m_labels_offset_dataspace);

  //  Look up actual word indices
  hsize_t offset(label_info.offset);
  hsize_t length(label_info.length);

  CHECK(label_info.offset >= 0 && label_info.length >= 0);

  m_dyn_memspace->setExtentSimple(1, &length, NULL);
  m_labels_wordseq_dataspace.selectHyperslab(H5S_SELECT_SET, &length, &offset);

  std::unique_ptr<int[]> wordseq(new int[label_info.length]);    // dynamic memory space for a word sequence
  m_labels_wordseq_dataset->read(wordseq.get(), H5::PredType::NATIVE_INT, *m_dyn_memspace, m_labels_wordseq_dataspace);
  
  // put words into the output container
  word_seq.assign(wordseq.get(), wordseq.get() + label_info.length);
*/
  word_seq.assign(m_label_descriptions[label_idx]->begin(), m_label_descriptions[label_idx]->end());
}

void BioASQHDF5Reader::getLabelsOfDocument(const long long doc_idx, std::vector<int>& labelset)
{
  if(labelset.size() > 0)
    labelset.clear();

  hsize_t doc_offset(doc_idx);
  hsize_t num_elements(1);

  OffsetInformation doc_info;
  // Select a hyperslab in extended portion of the dataset.
  m_doc_label_pair_offset_dataspace.selectHyperslab(H5S_SELECT_SET, &num_elements, &doc_offset);
  m_doc_label_pair_offset_dataset->read(&doc_info, *offset_type, *m_fixed_one_memspace, m_documents_offset_dataspace);

  //  Look up actual word indices
  hsize_t offset(doc_info.offset);
  hsize_t length(doc_info.length);

  CHECK(doc_info.offset >= 0 && doc_info.length >= 0);

  m_dyn_memspace->setExtentSimple(1, &length, NULL);
  m_doc_label_pair_lseq_dataspace.selectHyperslab(H5S_SELECT_SET, &length, &offset);

  std::unique_ptr<int[]> labels(new int[doc_info.length]);    // dynamic memory space for a word sequence
  m_doc_label_pair_lseq_dataset->read(labels.get(), H5::PredType::NATIVE_INT, *m_dyn_memspace, m_doc_label_pair_lseq_dataspace);
  
  // put words into the output container
  labelset.assign(labels.get(), labels.get() + doc_info.length);
}

// Getter
int BioASQHDF5Reader::getNumOfWords() const
{
  return m_num_words;
}

int BioASQHDF5Reader::getNumOfLabels() const
{
  return m_num_labels;
}

int BioASQHDF5Reader::getNumOfSeenLabels() const
{
  return m_num_seen_labels;
}

int BioASQHDF5Reader::getNumOfTrainInstances() const
{
  return m_num_train_instances;
}

int BioASQHDF5Reader::getNumOfValidInstances() const
{
  return m_num_valid_instances;
}

int BioASQHDF5Reader::getNumOfTestInstances() const
{
  return m_num_test_instances;
}

void BioASQHDF5Reader::getTrainInstance(const long long inst_idx, std::vector<int>& words, std::vector<int>& labels)
{
  //const long long idx = m_train_instance_indices[inst_idx].second;
  const long long idx = m_train_instance_offset + inst_idx;
  getWordSeqFromDocument(idx, words);
  getLabelsOfDocument(idx, labels);
}

void BioASQHDF5Reader::getValidInstance(const long long inst_idx, std::vector<int>& words, std::vector<int>& labels)
{
  //const long long idx = m_valid_instance_indices[inst_idx].second;
  const long long idx = m_valid_instance_offset + inst_idx;
  getWordSeqFromDocument(idx, words);
  getLabelsOfDocument(idx, labels);
}

void BioASQHDF5Reader::getTestInstance(const long long inst_idx, std::vector<int>& words, std::vector<int>& labels)
{
  //const long long idx = m_test_instance_indices[inst_idx].second;
  const long long idx = m_test_instance_offset + inst_idx;
  getWordSeqFromDocument(idx, words);
  getLabelsOfDocument(idx, labels);
}

void BioASQHDF5Reader::getTrainInstances(const long long base_idx,
                                        const int num_elements_,
                                        std::vector<std::vector<int> >& words_in_docs,
                                        std::vector<std::vector<int> >& labels_in_docs)
{
  hsize_t base_offset(base_idx);
  hsize_t num_elements(num_elements_);

  {
    std::unique_ptr<OffsetInformation[]> docs_info(new OffsetInformation[num_elements]);
    H5::DataSpace docs_memspace(1, &num_elements, NULL);

    // Select a hyperslab in extended portion of the dataset.
    m_documents_offset_dataspace.selectHyperslab(H5S_SELECT_SET, &num_elements, &base_offset);
    m_documents_offset_dataset->read(docs_info.get(), *offset_type, docs_memspace, m_documents_offset_dataspace);

    //  Look up actual word indices
    hsize_t start_word_offset = docs_info.get()[0].offset;
    hsize_t total_length = docs_info.get()[num_elements-1].offset - docs_info.get()[0].offset + docs_info.get()[num_elements-1].length;

    CHECK(start_word_offset >= 0 && total_length >= 0);

    m_dyn_memspace->setExtentSimple(1, &total_length, NULL);
    m_documents_wordseq_dataspace.selectHyperslab(H5S_SELECT_SET, &total_length, &start_word_offset);

    std::unique_ptr<int[]> wordseq(new int[total_length]);    // dynamic memory space for a word sequence
    m_documents_wordseq_dataset->read(wordseq.get(), H5::PredType::NATIVE_INT, *m_dyn_memspace, m_documents_wordseq_dataspace);
    
    // put words into the output container
    size_t s_idx = 0, e_idx=0;
    for(size_t i=0; i < num_elements; i++)
    {
      words_in_docs[i].clear();
      e_idx = s_idx + docs_info[i].length;
      words_in_docs[i].assign(wordseq.get()+s_idx, wordseq.get()+e_idx);
      s_idx = e_idx;
    }

    H5::DataSet::vlenReclaim(docs_info.get(), *offset_type, docs_memspace);
    //test_memspace.close();
  }

  {
    std::unique_ptr<OffsetInformation[]> labels_info(new OffsetInformation[num_elements]);
    H5::DataSpace labels_memspace(1, &num_elements, NULL);

    // Select a hyperslab in extended portion of the dataset.
    m_doc_label_pair_offset_dataspace.selectHyperslab(H5S_SELECT_SET, &num_elements, &base_offset);
    m_doc_label_pair_offset_dataset->read(labels_info.get(), *offset_type, labels_memspace, m_doc_label_pair_offset_dataspace);

    hsize_t start_label_offset = labels_info.get()[0].offset;
    hsize_t total_length = labels_info.get()[num_elements-1].offset - labels_info.get()[0].offset + labels_info.get()[num_elements-1].length;

    CHECK(start_label_offset >= 0 && total_length >= 0);

    m_dyn_memspace->setExtentSimple(1, &total_length, NULL);
    m_doc_label_pair_lseq_dataspace.selectHyperslab(H5S_SELECT_SET, &total_length, &start_label_offset);

    std::unique_ptr<int[]> labelseq(new int[total_length]);    // dynamic memory space for a word sequence
    m_doc_label_pair_lseq_dataset->read(labelseq.get(), H5::PredType::NATIVE_INT, *m_dyn_memspace, m_doc_label_pair_lseq_dataspace);
    
    // put words into the output container
    size_t s_idx = 0, e_idx=0;
    for(size_t i=0; i < num_elements; i++)
    {
      labels_in_docs[i].clear();
      e_idx = s_idx + labels_info[i].length;
      labels_in_docs[i].assign(labelseq.get()+s_idx, labelseq.get()+e_idx);
      s_idx = e_idx;
    }

    H5::DataSet::vlenReclaim(labels_info.get(), *offset_type, labels_memspace);
  }
}

void BioASQHDF5Reader::getValidInstances(const long long base_idx,
                                        const int num_elements_,
                                        std::vector<std::vector<int> >& words_in_docs,
                                        std::vector<std::vector<int> >& labels_in_docs)
{
  //const long long valid_base_idx = m_valid_instance_indices[base_idx].second;
  const long long valid_base_idx = m_valid_instance_offset + base_idx;

  hsize_t base_offset(valid_base_idx);
  hsize_t num_elements(num_elements_);

  {
    std::unique_ptr<OffsetInformation[]> docs_info(new OffsetInformation[num_elements]);
    H5::DataSpace docs_memspace(1, &num_elements, NULL);

    // Select a hyperslab in extended portion of the dataset.
    m_documents_offset_dataspace.selectHyperslab(H5S_SELECT_SET, &num_elements, &base_offset);
    m_documents_offset_dataset->read(docs_info.get(), *offset_type, docs_memspace, m_documents_offset_dataspace);

    //  Look up actual word indices
    hsize_t start_word_offset = docs_info.get()[0].offset;
    hsize_t total_length = docs_info.get()[num_elements-1].offset - docs_info.get()[0].offset + docs_info.get()[num_elements-1].length;

    CHECK(start_word_offset >= 0 && total_length >= 0);

    m_dyn_memspace->setExtentSimple(1, &total_length, NULL);
    m_documents_wordseq_dataspace.selectHyperslab(H5S_SELECT_SET, &total_length, &start_word_offset);

    std::unique_ptr<int[]> wordseq(new int[total_length]);    // dynamic memory space for a word sequence
    m_documents_wordseq_dataset->read(wordseq.get(), H5::PredType::NATIVE_INT, *m_dyn_memspace, m_documents_wordseq_dataspace);
    
    // put words into the output container
    size_t s_idx = 0, e_idx=0;
    for(size_t i=0; i < num_elements; i++)
    {
      words_in_docs[i].clear();
      e_idx = s_idx + docs_info[i].length;
      words_in_docs[i].assign(wordseq.get()+s_idx, wordseq.get()+e_idx);
      s_idx = e_idx;
    }

    H5::DataSet::vlenReclaim(docs_info.get(), *offset_type, docs_memspace);
    //test_memspace.close();
  }

  {
    std::unique_ptr<OffsetInformation[]> labels_info(new OffsetInformation[num_elements]);
    H5::DataSpace labels_memspace(1, &num_elements, NULL);

    // Select a hyperslab in extended portion of the dataset.
    m_doc_label_pair_offset_dataspace.selectHyperslab(H5S_SELECT_SET, &num_elements, &base_offset);
    m_doc_label_pair_offset_dataset->read(labels_info.get(), *offset_type, labels_memspace, m_doc_label_pair_offset_dataspace);

    hsize_t start_label_offset = labels_info.get()[0].offset;
    hsize_t total_length = labels_info.get()[num_elements-1].offset - labels_info.get()[0].offset + labels_info.get()[num_elements-1].length;

    CHECK(start_label_offset >= 0 && total_length >= 0);

    m_dyn_memspace->setExtentSimple(1, &total_length, NULL);
    m_doc_label_pair_lseq_dataspace.selectHyperslab(H5S_SELECT_SET, &total_length, &start_label_offset);

    std::unique_ptr<int[]> labelseq(new int[total_length]);    // dynamic memory space for a word sequence
    m_doc_label_pair_lseq_dataset->read(labelseq.get(), H5::PredType::NATIVE_INT, *m_dyn_memspace, m_doc_label_pair_lseq_dataspace);
    
    // put words into the output container
    size_t s_idx = 0, e_idx=0;
    for(size_t i=0; i < num_elements; i++)
    {
      labels_in_docs[i].clear();
      e_idx = s_idx + labels_info[i].length;
      labels_in_docs[i].assign(labelseq.get()+s_idx, labelseq.get()+e_idx);
      s_idx = e_idx;
    }

    H5::DataSet::vlenReclaim(labels_info.get(), *offset_type, labels_memspace);
  }
}

void BioASQHDF5Reader::getTestInstances(const long long base_idx,
                                        const int num_elements_,
                                        std::vector<std::vector<int> >& words_in_docs,
                                        std::vector<std::vector<int> >& labels_in_docs)
{
  //const long long test_base_idx = m_test_instance_indices[base_idx].second;
  const long long test_base_idx = m_test_instance_offset + base_idx;

  hsize_t base_offset(test_base_idx);
  hsize_t num_elements(num_elements_);

  {
    std::unique_ptr<OffsetInformation[]> docs_info(new OffsetInformation[num_elements]);
    H5::DataSpace docs_memspace(1, &num_elements, NULL);

    // Select a hyperslab in extended portion of the dataset.
    m_documents_offset_dataspace.selectHyperslab(H5S_SELECT_SET, &num_elements, &base_offset);
    m_documents_offset_dataset->read(docs_info.get(), *offset_type, docs_memspace, m_documents_offset_dataspace);

    //  Look up actual word indices
    hsize_t start_word_offset = docs_info.get()[0].offset;
    hsize_t total_length = docs_info.get()[num_elements-1].offset - docs_info.get()[0].offset + docs_info.get()[num_elements-1].length;

    CHECK(start_word_offset >= 0 && total_length >= 0);

    m_dyn_memspace->setExtentSimple(1, &total_length, NULL);
    m_documents_wordseq_dataspace.selectHyperslab(H5S_SELECT_SET, &total_length, &start_word_offset);

    std::unique_ptr<int[]> wordseq(new int[total_length]);    // dynamic memory space for a word sequence
    m_documents_wordseq_dataset->read(wordseq.get(), H5::PredType::NATIVE_INT, *m_dyn_memspace, m_documents_wordseq_dataspace);
    
    // put words into the output container
    size_t s_idx = 0, e_idx=0;
    for(size_t i=0; i < num_elements; i++)
    {
      words_in_docs[i].clear();
      e_idx = s_idx + docs_info[i].length;
      words_in_docs[i].assign(wordseq.get()+s_idx, wordseq.get()+e_idx);
      s_idx = e_idx;
    }

    H5::DataSet::vlenReclaim(docs_info.get(), *offset_type, docs_memspace);
    //test_memspace.close();
  }

  {
    std::unique_ptr<OffsetInformation[]> labels_info(new OffsetInformation[num_elements]);
    H5::DataSpace labels_memspace(1, &num_elements, NULL);

    // Select a hyperslab in extended portion of the dataset.
    m_doc_label_pair_offset_dataspace.selectHyperslab(H5S_SELECT_SET, &num_elements, &base_offset);
    m_doc_label_pair_offset_dataset->read(labels_info.get(), *offset_type, labels_memspace, m_doc_label_pair_offset_dataspace);

    hsize_t start_label_offset = labels_info.get()[0].offset;
    hsize_t total_length = labels_info.get()[num_elements-1].offset - labels_info.get()[0].offset + labels_info.get()[num_elements-1].length;

    CHECK(start_label_offset >= 0 && total_length >= 0);

    m_dyn_memspace->setExtentSimple(1, &total_length, NULL);
    m_doc_label_pair_lseq_dataspace.selectHyperslab(H5S_SELECT_SET, &total_length, &start_label_offset);

    std::unique_ptr<int[]> labelseq(new int[total_length]);    // dynamic memory space for a word sequence
    m_doc_label_pair_lseq_dataset->read(labelseq.get(), H5::PredType::NATIVE_INT, *m_dyn_memspace, m_doc_label_pair_lseq_dataspace);
    
    // put words into the output container
    size_t s_idx = 0, e_idx=0;
    for(size_t i=0; i < num_elements; i++)
    {
      labels_in_docs[i].clear();
      e_idx = s_idx + labels_info[i].length;
      labels_in_docs[i].assign(labelseq.get()+s_idx, labelseq.get()+e_idx);
      s_idx = e_idx;
    }

    H5::DataSet::vlenReclaim(labels_info.get(), *offset_type, labels_memspace);
  }
}

// Vocabulary accessors
const std::string& BioASQHDF5Reader::getWordNameOfIndex(int index)
{
  CHECK(index >= 0 && index < m_num_words) << "Out of range: " << index;
  return m_word_name_freq_pairs[index].first;
}

int BioASQHDF5Reader::getWordFrequencyOfIndex(int index)
{
  CHECK(index >= 0 && index < m_num_words) << "Out of range: " << index;
  return m_word_name_freq_pairs[index].second;
}

const std::string& BioASQHDF5Reader::getLabelNameOfIndex(int index)
{
  CHECK(index >= 0 && index < m_num_labels) << "Out of range: " << index;
  return m_label_names[index];
}
