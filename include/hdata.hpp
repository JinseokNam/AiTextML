#ifndef HDATA_HPP_
#define HDATA_HPP_

#include <string>
#include <boost/shared_ptr.hpp>
#include "H5Cpp.h"

class EntityInformation
{
public:
  EntityInformation(std::string name_, int freq_, int index_)
    : name(name_), freq(freq_), index(index_)
  {
  }

  std::string name;
  int freq;
  int index;
};

typedef boost::shared_ptr<EntityInformation> entity_ptr;

typedef struct {
  char* name;
  unsigned long long offset;
  int length;     
} OffsetInformation;

typedef struct {
  char* name;
  int freq;
} FrequencyInformation;

class BioASQHDF5Data {
private:
  BioASQHDF5Data(){}
public:
  static const H5std_string SPLIT_GROUP_NAME;
  static const H5std_string TRAIN_SPLIT_DATASET_NAME;
  static const H5std_string VALID_SPLIT_DATASET_NAME;
  static const H5std_string TEST_SPLIT_DATASET_NAME;

  static const H5std_string WORDS_GROUP_NAME;
  static const H5std_string DOCUMENTS_GROUP_NAME;
  static const H5std_string LABELS_GROUP_NAME;
  static const H5std_string DOC_LABEL_PAIR_GROUP_NAME;

  static const H5std_string NUM_INSTANCES_ATTR_NAME;
  static const H5std_string NUM_WORDS_ATTR_NAME;
  static const H5std_string NUM_LABELS_ATTR_NAME;
  static const H5std_string NUM_SEEN_LABELS_ATTR_NAME;

  static const H5std_string ENTITY_INDICES_DATASET_NAME;
  static const H5std_string OFFSET_INFO_DATASET_NAME;
  static const H5std_string FREQUENCY_DATASET_NAME;

  static const std::string member_name;
  static const std::string member_offset;
  static const std::string member_length;
  static const std::string member_freq;

  static const std::string DIR_SEP;
};

#endif
