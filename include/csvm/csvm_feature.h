#ifndef CSVM_FEATURE_H
#define CSVM_FEATURE_H

#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>

using namespace std;
namespace csvm{
  
  class Feature{
    
    
  public:
      vector< double> content;
      string label;
      int size;
      Feature(int size,double initValue);
      Feature(Feature* f);
      double getDistanceSq(Feature* f);
  };
  
  
  
}

#endif