#ifndef CSVM_FEATURE_H
#define CSVM_FEATURE_H

#include <vector>
#include <string>
using namespace std;
namespace csvm{
  
  class Feature{
    
    
  public:
      vector< double> content;
      string label;
      int size;
      Feature(int size,double initValue);

    
  };
  
  
  
}

#endif