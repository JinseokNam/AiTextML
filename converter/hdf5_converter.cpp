#include <iostream>
#include <glog/logging.h>
#include "boost/program_options.hpp" 
#include "ohdata.hpp"
#include "H5Cpp.h"

int main(int argc, char* argv[])
{
  google::InitGoogleLogging(argv[0]);

  std::string traindata_filepath;
  std::string trainlabel_filepath;
  std::string validdata_filepath;
  std::string validlabel_filepath;
  std::string testdata_filepath;
  std::string testlabel_filepath;
  std::string label_description_filepath;
  std::string word_vocabulary;
  std::string seen_label_vocabulary;
  std::string total_label_vocabulary;
  std::string output_filepath;

  int word_min_count(0), label_min_count(0);

  boost::program_options::options_description desc("Options");
  desc.add_options()
    ("help","Display help messages")
    ("traindata_path", boost::program_options::value<std::string>(&traindata_filepath)->required(), "Filepath of the training corpus")
    ("trainlabel_path", boost::program_options::value<std::string>(&trainlabel_filepath)->required(), "Filepath of the training label set")
    ("validdata_path", boost::program_options::value<std::string>(&validdata_filepath), "Filepath of the validing corpus")
    ("validlabel_path", boost::program_options::value<std::string>(&validlabel_filepath), "Filepath of the validing label set")
    ("testdata_path", boost::program_options::value<std::string>(&testdata_filepath), "Filepath of the testing corpus")
    ("testlabel_path", boost::program_options::value<std::string>(&testlabel_filepath), "Filepath of the testing label set")
    ("label_description_path", boost::program_options::value<std::string>(&label_description_filepath)->required(), "Filepath of label descriptions")
    ("word_vocabulary", boost::program_options::value<std::string>(&word_vocabulary)->required(), "Filepath of the unique word set")
    ("word_min_count", boost::program_options::value<int>(&word_min_count)->default_value(5), "Minimum count of word frequency")
    ("seen_label_vocabulary", boost::program_options::value<std::string>(&seen_label_vocabulary)->required(), "Filepath of the unique label set")
    ("total_label_vocabulary", boost::program_options::value<std::string>(&total_label_vocabulary)->required(), "Filepath of the unique label set")
    ("label_min_count", boost::program_options::value<int>(&label_min_count)->default_value(1), "Minimum count of label frequency")
    ("output_filepath", boost::program_options::value<std::string>(&output_filepath)->required(), "Filepath to store the HDF5 file");

  boost::program_options::variables_map vm;
  try
  {
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);

    if(vm.count("help"))
    {
      std::cout << "Basic Command Line Parameter App" << std::endl
                << desc << std::endl;
      return 0;
    }

    boost::program_options::notify(vm);
  }
  catch(boost::program_options::error& e)
  {
    std::cerr << "Option parsing error: " << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
    return 1;
  }
 
  BioASQHDF5Writer bw(traindata_filepath, trainlabel_filepath, validdata_filepath, validlabel_filepath, testdata_filepath, testlabel_filepath, label_description_filepath, output_filepath);
  bw.build_word_vocabulary(word_vocabulary, word_min_count);
  bw.build_label_vocabulary(seen_label_vocabulary, label_min_count);
  bw.build_label_vocabulary(total_label_vocabulary, label_min_count);
  bw.convert2hdf5();

  std::cout << "DONE" << std::endl;

  return 0;
}
