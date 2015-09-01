//copyright Marc Groefsema (c) 2015

#ifndef HB_HOG_DESCRIPTOR_H
#define HB_HOG_DESCRIPTOR_H


#include <vector>
#include <iostream>

namespace hogbovw{

   class{
      int nBins; //number of angular orientated histogram bins bins
      int cellSize; // assumes square cell
      int cellStride; //the steps the cell make across the input
      
      
      
   public:
      
      vector<float> getHOG();
      
   }HOGDescriptor;


}


#endif
