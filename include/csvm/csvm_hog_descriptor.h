
#ifndef CSVM_HOG_DESCRIPTOR_H
#define CSVM_HOG_DESCRIPTOR_H


#include <vector>
#include <iostream>
#include <cmath>
#include "csvm_patch.h"
#include "csvm_image.h"
#include "csvm_feature.h"

using namespace std;


namespace csvm{

	struct HOGSettings {
		int nBins;                //number of angular orientated histogram bins bins
		int cellSize;             // assumes square cell
		int cellStride;           //the steps the cell make across the input
		int blockSize;			
		int numberOfCells;
		bool useGreyPixel;

	};
   class HOGDescriptor{
	   HOGSettings settings;

   public:
	   HOGDescriptor();
      //HOGDescriptor(int nBins, int cellSize, int blockSize, bool useGreyPixel); 
	  //HOGDescriptor(int nBins, int numberOfCells, int blockSize, bool useGreyPixel);
	  Feature getHOG(Patch patch,int channel, bool useGreyPixel);

   private:
	   double computeXGradient(Patch patch, int x, int y);
	   double computeYGradient(Patch patch, int x, int y);
	   double computeMagnitude(double x, double y);
	   double computeOrientation(double x, double y);
      
   };


}


#endif
