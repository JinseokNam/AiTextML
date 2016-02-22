#include "utils.hpp"
#include "labeled_corpus.hpp"

std::string LabeledCorpus::get_label_filename() const
{
  return m_label_filename;
}

std::string LabeledCorpus::get_label_desc_filename() const
{
  return m_label_desc_filename;
}

long long LabeledCorpus::get_num_labels() const
{
  return (long long) m_label_map.size();
}

void LabeledCorpus::build_label_dictionary()
{
  std::string label_line;
  std::ifstream label_file;

  if(get_label_filename().empty())  return;   // no label information given

  label_file.open(this->get_label_filename());

  CHECK(label_file.is_open()) << "Failed to open the file: " << this->get_label_filename();

  std::map<std::string, int> temp_label_dictionary;
  
  while(std::getline(label_file, label_line))
  {
    std::vector<std::string> labels = split(label_line.c_str(), ' ');
    for(const std::string &label : labels)
    {
      std::map<std::string, int>::iterator map_it = temp_label_dictionary.find(label);
      if(map_it == temp_label_dictionary.end()) temp_label_dictionary[label] = 0;
      temp_label_dictionary[label] += 1;
    }
  }

  std::vector<std::pair<std::string, int> > v;
  std::copy(temp_label_dictionary.begin(), temp_label_dictionary.end(), std::back_inserter<std::vector<std::pair<std::string, int> > > (v));
  auto value_cmp = [](std::pair<std::string,int> const & a, std::pair<std::string,int> const & b) 
  { 
    // desending
    return a.second > b.second;
  };
  std::sort(v.begin(), v.end(), value_cmp);

  for(size_t i=0; i < v.size(); i++)
  {
    //std::cout << v[i].first << std::endl;   // outputs label names ordered
    m_label_map[v[i].first] = i;
    m_inv_label_map[i] = v[i].first;
  }

  if(!m_label_desc_filename.empty())
  {
    std::ifstream label_desc_file;
    label_desc_file.open(this->get_label_desc_filename());
    CHECK(label_desc_file.is_open()) << "Failed to open the file: " << this->get_label_desc_filename();

    const std::string delimiter = ":::";
    while(std::getline(label_desc_file, label_line))
    {
      std::string label_name = label_line.substr(0, label_line.find(delimiter));
      std::string label_desc = label_line.substr(label_line.find(delimiter)+delimiter.length(),label_line.length());

      m_label_desc_map[label_name] = label_desc;
    }
    LOG(INFO) << m_label_desc_map.size() << " descriptions.";
  }
}

std::map<std::string, long long>& LabeledCorpus::getLabelDictionary()
{
  return m_label_map;
}

std::map<long long, std::string>& LabeledCorpus::getInvLabelDictionary()
{
  return m_inv_label_map;
}

std::map<std::string, std::string>& LabeledCorpus::getLabelDescDictionary()
{
  return m_label_desc_map;
}
