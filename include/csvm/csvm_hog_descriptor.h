
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
   enum Colour {           //used for accessing the right indexes in dynamic array
      RED = 0,
      GREEN = 1,
      BLUE = 2,
     GRAY = 3,
   };

   enum Padding {          //type of padding used when comoputing the gradient and magnitude along boundary
      ZERO = 0,               //ZERO means it will be padded with make-believe pixels with value 0
      IDENTITY = 1,           //IDENTITY means the make-believe pixels will get the same values as those at the boundary
      NONE = 2,               //means we perform no padding, and thus only compute magnitudes and orientations of the pixels that we can, THIS RESULTS IN A MORE ACCURATE, BUT SPARSER REPRESENTATION OF THE PATCH!
   };

   enum INTERPOLATION {
      INTERPOLATE_BINARY = 0, //the values are binned directly in the bin they fall in
      INTERPOLATE_LINEAR = 1, //the values are proportionally binned to the two nearest bins that it falls between.
      //INTERPOLATE_TRILINEAR = 2,
   };

   enum HOGGRADIENT {
      MAGNITUDE = 0,
      ORIENTATION = 1,
   };
   
   enum BINNING {
      CROSSCOLOUR = 0,     //here only orientations and magnitudes are comoputed by every colour channel, but they are binned to the same HOG of the cell. 
      BYCOLOUR = 1,     //meaning that every colour channel will get its of HOG feature, and the feature vector becomes 3 x as long (when using colours)
   };

   enum POSTPROCESSING {      //should we perform any data-transformation on the final feature vector? (NOT CURRENTLY IN USE!)
      PURE = 0,            //no postprocessing
      STANDARDISATION = 1,    //yes, standardisation
      NORMALISATION = 2,      //yes, normalisation
      LTWONORM = 3,
      CLIPNORM = 4, //yes, clipping as in paper http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
   };

   struct HOGSettings {
      unsigned int nBins;                //number of angular orientated bins which make up the histogram of magnitudes, range: 6-36(ish). try to use multiples of 9 or 6
      unsigned int cellSize;             // assumes square cell. Best to make it an clean divisor of patchsize
      unsigned int cellStride;           // the stride the cell window makes when iterating over the patch. (This may also be the cell size itself for a seperation into quadrants)
      unsigned int patchSize;          //the size of the patch       
      //unsigned int numberOfCells;         //is internally computed by virtue of cell size, stride, and blocksize.
      bool useColourPixel;               //use the gray pixels? or all color channels. If all channels, then feature is 3 times as large. Default is true
      Padding padding;                 // what type of padding should be used to deal with 
     INTERPOLATION interpol;        //what type of interpolation should be performed?
     BINNING binmethod;          //how do we bin over colour channels?
     POSTPROCESSING postproccess;      //do we want any normalisation or standardisation?
     int debugLevel;
   };
   class HOGDescriptor{
      HOGSettings settings;

   public:
      bool debugOut, normalOut;
      HOGDescriptor();
      void setSettings(HOGSettings s);
      Feature getHOG(Patch& patch); //,int channel, bool useGreyPixel);

   private:
      double computeXGradient(Patch patch, int x, int y, Colour col);   //to compute the x gradient
      double computeYGradient(Patch patch, int x, int y, Colour col);   //self-explanatry
      double computeMagnitude(double x, double y);             //also self explanatory
      double computeOrientation(double x, double y);           //~
     void binPixel(size_t X, size_t Y, Colour col, vector<double>& cellOrientationHistogram, Patch& block);       //bins the magnitude of a pixel (X,Y) in colour channel col into cellOrientationHistogram, 
     void binPixel(size_t X, size_t Y, Colour col, vector<double>& cellOrientationHistogram, double ****imageTranspose);   //does the same, but uses the transpose, thus avoiding repeated computing of magnitudes etc. 
     //void binTranspose(double imageTranspose[], Patch& block);
     double ****patchTranspose(Patch& block, double ****imageTranspose, unsigned int colours);        //computes the entire gradient-transposition of a patch
     vector <double> computeCellHOG(double ****imageTranspose, unsigned int cellX, unsigned int cellY);  //computes the HOG for a given cell.
     vector <double> postProcess(vector <double> blockHistogram);
   };


}


#endif
