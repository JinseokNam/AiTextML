#include "hdata.hpp"

const H5std_string BioASQHDF5Data::SPLIT_GROUP_NAME( "Split" );
const H5std_string BioASQHDF5Data::TRAIN_SPLIT_DATASET_NAME( "Trainset" );
const H5std_string BioASQHDF5Data::VALID_SPLIT_DATASET_NAME( "Validset" );
const H5std_string BioASQHDF5Data::TEST_SPLIT_DATASET_NAME( "Testset" );

const H5std_string BioASQHDF5Data::WORDS_GROUP_NAME( "Words" );
const H5std_string BioASQHDF5Data::DOCUMENTS_GROUP_NAME( "Documents" );
const H5std_string BioASQHDF5Data::LABELS_GROUP_NAME( "Labels" );
const H5std_string BioASQHDF5Data::DOC_LABEL_PAIR_GROUP_NAME( "Document_Label_Pair" );

const H5std_string BioASQHDF5Data::NUM_INSTANCES_ATTR_NAME( "num_instances" );
const H5std_string BioASQHDF5Data::NUM_WORDS_ATTR_NAME( "num_words" );
const H5std_string BioASQHDF5Data::NUM_LABELS_ATTR_NAME( "num_labels" );
const H5std_string BioASQHDF5Data::NUM_SEEN_LABELS_ATTR_NAME( "num_seen_labels" );

const H5std_string BioASQHDF5Data::ENTITY_INDICES_DATASET_NAME( "entity_indices" );
const H5std_string BioASQHDF5Data::OFFSET_INFO_DATASET_NAME( "offset_info" );
const H5std_string BioASQHDF5Data::FREQUENCY_DATASET_NAME( "freq_info" );

const std::string BioASQHDF5Data::member_name( "Name" );
const std::string BioASQHDF5Data::member_offset( "Offset" );
const std::string BioASQHDF5Data::member_length( "Length" );
const std::string BioASQHDF5Data::member_freq( "Freq" );

const std::string BioASQHDF5Data::DIR_SEP( "/" );
