#ifndef CSVM_FEATURE_H
#define CSVM_FEATURE_H

/* This Feature class is a bit like a vector, except that it also contains a label.
 * Also, it has some often used distance functions with it 
 * 
 * 
 */

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
      Feature(vector<double>& vect);
      void setLabelId(int id);
      unsigned int getLabelId();
      double getDistanceSq(Feature& f);
      double getManhDist(Feature* f);
  };
  
  
  
}

#endif