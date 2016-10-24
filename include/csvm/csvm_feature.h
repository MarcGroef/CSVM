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
     bool debugOut, normalOut;
      vector< float> content;
      string label;
      int labelId;
      int size;
      int squareId;
      unsigned int getSquareId();
      void setSquareId(int id);
      Feature(int size,float initValue);
      Feature(Feature* f);
      Feature(vector<float>& vect);
      void setLabelId(int id);
      unsigned int getLabelId();
      float getDistanceSq(Feature& f);
      float getManhDist(Feature* f);
  };
  
  
  
}

#endif
