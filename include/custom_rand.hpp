#ifndef CUSTOM_RAND_HPP_
#define CUSTOM_RAND_HPP_

#include "common.hpp"
 
class UniformRandom
{
private:
  boost::variate_generator<boost::mt19937&, boost::random::uniform_real_distribution<>> *generator;
 
public:
  UniformRandom( boost::mt19937 &rng, float min_, float max_)
  {
    generator = new boost::variate_generator<boost::mt19937&, boost::random::uniform_real_distribution<>>(
                  rng,
                  boost::random::uniform_real_distribution<double>( min_, max_ ));
  }

  ~UniformRandom()
  {
    delete generator;
  }

  float operator()()
  {
    return (*generator)();
  }
};


class NormalRandom
{
private:
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<>> *generator;
 
public:
  NormalRandom( boost::mt19937 &rng, float mean_, float std_)
  {
    generator = new boost::variate_generator<boost::mt19937&, boost::normal_distribution<>>(
                  rng, 
                  boost::normal_distribution<>( mean_, std_ ));
  }

  ~NormalRandom()
  {
    delete generator;
  }

  float operator()()
  {
    return (*generator)();
  }
};

class IntRandom
{
private:
  boost::variate_generator<boost::mt19937&, boost::random::uniform_int_distribution<>> *generator;
 
public:
  IntRandom( boost::mt19937& rng, int min_, int max_)
  {
    generator = new boost::variate_generator<boost::mt19937&, boost::random::uniform_int_distribution<>>(
                  rng,
                  boost::random::uniform_int_distribution<>( min_, max_ ));
  }

  ~IntRandom()
  {
    delete generator;
  }

  int operator()()
  {
    return (*generator)();
  }
};
#endif
