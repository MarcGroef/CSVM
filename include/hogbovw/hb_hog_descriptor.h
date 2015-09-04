
#ifndef HB_HOG_DESCRIPTOR_H
#define HB_HOG_DESCRIPTOR_H


#include <vector>
#include <iostream>
#include <cmath>

#include "hb_opencv_incl.h"

using namespace std;
using namespace cv;

namespace hogbovw{

   class HOGDescriptor{
      int nBins; //number of angular orientated histogram bins bins
      int cellSize; // assumes square cell
      int cellStride; //the steps the cell make across the input
      int blockSize;
      int blockStride;
      
   public:
      HOGDescriptor(int nBins,int cellSize,int cellStride,int blockSize,int blockStride);
      vector< vector< float> > getHOG(Mat image);
      
   };


}


#endif
