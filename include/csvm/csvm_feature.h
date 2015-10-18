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
      int labelId;
      int size;
      Feature(int size,double initValue);
      Feature(Feature* f);
      
      void setLabelId(int id);
      int getLabelId();
      double getDistanceSq(Feature& f);
      double getManhDist(Feature* f);
  };
  
  
  
}

#endif