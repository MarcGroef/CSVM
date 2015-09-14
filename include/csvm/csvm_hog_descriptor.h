
#ifndef CSVM_HOG_DESCRIPTOR_H
#define CSVM_HOG_DESCRIPTOR_H


#include <vector>
#include <iostream>
#include <cmath>

#include "csvm_image.h"

using namespace std;


namespace csvm{

   class HOGDescriptor{
      int nBins;                //number of angular orientated histogram bins bins
      int cellSize;             // assumes square cell
      int cellStride;           //the steps the cell make across the input
      int blockSize;
      int blockStride;
      
   public:
      HOGDescriptor(int nBins, int cellSize, int cellStride, int blockSize, int blockStride); 
      vector< vector< double> > getHOG(Image image,int channel);
      
   };


}


#endif
