#ifndef MEASURES_HPP_
#define MEASURES_HPP_

#include "common.hpp"

typedef float real;

class Measures
{
public:
  static real rankloss(std::set<int>& targets, real *sim_scores, long long L)
  {
    long long incorrect_pairwise_ranking = 0;
    for(const auto& pos_label : targets)
    {
      const real& pos_label_score = sim_scores[pos_label];
      for(long long i = 0; i < L; i++)
      {
        const bool is_pos_label = targets.find(i) != targets.end();
        if(!is_pos_label)   
        {
          const real& neg_label_score = sim_scores[i];
          if(pos_label_score < neg_label_score)
          {
            incorrect_pairwise_ranking++;
          }
        }
      }
    } 
    return incorrect_pairwise_ranking/(real)(targets.size() * (L-targets.size()));
  }
};

#endif

