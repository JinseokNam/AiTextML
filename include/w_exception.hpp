#ifndef W_EXCEPTION_HPP_
#define W_EXCEPTION_HPP_

#include <exception>

class FileNotFoundException : public std::exception
{
private:
  std::string s;
public:
  FileNotFoundException(std::string ss) : s(ss) {}
  ~FileNotFoundException() throw() {}
  const char* what() const throw() {return s.c_str();}
};

#endif
