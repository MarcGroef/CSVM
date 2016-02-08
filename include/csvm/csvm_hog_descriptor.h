
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
   enum Colour {
      RED = 0,
      GREEN = 1,
      BLUE = 2,
	  GRAY = 3,
   };

   enum Padding {
      ZERO = 0,                  //the type of padding used
      IDENTITY = 1,
      NONE = 2,
   };

   enum INTERPOLATION {
	   INTERPOLATE_BINARY = 0,
	   INTERPOLATE_LINEAR = 1,
	   //INTERPOLATE_TRILINEAR = 2,
   };

   enum HOGGRADIENT {
	   MAGNITUDE = 0,
	   ORIENTATION = 1,
   };

   struct HOGSettings {
      unsigned int nBins;                //number of angular orientated bins which make up the histogram of magnitudes
      unsigned int cellSize;             // assumes square cell. Best to make it an even divisor of blocksize 
      unsigned int cellStride;           // the stride the cell window makes when iterating over the patch. (This may also be the cell size itself for a seperation into quadrants)
      int patchSize;          //the size of the patch       
      //unsigned int numberOfCells;         //is internally computed by virtue of cell size, stride, and blocksize.
      bool useGreyPixel;               //use the gray pixels? or all color channels. If all channels, then feature is 3 times as large. Default is true
      //bool interpolation;               //whether binning is proportionate (true) or direct (false). Default is false
      Padding padding;                 // what type of padding should be used to deal with 
	  INTERPOLATION interpol;
   };
   class HOGDescriptor{
      HOGSettings settings;

   public:
      bool debugOut, normalOut;
      HOGDescriptor();
      //HOGDescriptor(int cellSize, int cellStride, int blockSize);
      void setSettings(HOGSettings s);
      //HOGDescriptor(int nBins, int cellSize, int blockSize, bool useGreyPixel); 
     //HOGDescriptor(int nBins, int numberOfCells, int blockSize, bool useGreyPixel);
	  Feature getHOG(Patch& patch);	//,int channel, bool useGreyPixel);

   private:
      double computeXGradient(Patch patch, int x, int y, Colour col);
      double computeYGradient(Patch patch, int x, int y, Colour col);
      double computeMagnitude(double x, double y);
      double computeOrientation(double x, double y);
	  void binPixel(size_t X, size_t Y, Colour col, vector<double>& cellOrientationHistogram, Patch& block);
	  void binPixel(size_t X, size_t Y, Colour col, vector<double>& cellOrientationHistogram, double ****imageTranspose);
	  //void binTranspose(double imageTranspose[], Patch& block);
   };


}


#endif
