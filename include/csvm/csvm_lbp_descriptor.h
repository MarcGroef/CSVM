#ifndef CSVM_LBP_DESCRIPTOR_H
#define CSVM_LBP_DESCRIPTOR_H

//DEPRECATED

#include <vector>
#include <iostream>
#include <cmath>
#include "csvm_patch.h"
#include "csvm_image.h"
#include "csvm_feature.h"


#include <bitset>


using namespace std;


namespace csvm {

	enum LBPColour {           //used for accessing the right indexes in dynamic array
		LRED = 0,
		LGREEN = 1,
		LBLUE = 2,
		LGRAY = 3,
	};

	enum LBPPadding {          //type of padding used when comoputing the gradient and magnitude along boundary
		LZERO = 0,               //ZERO means it will be padded with make-believe pixels with value 0
		LIDENTITY = 1,           //IDENTITY means the make-believe pixels will get the same values as those at the boundary
		LNONE = 2,               //means we perform no padding, and thus only compute magnitudes and orientations of the pixels that we can, THIS RESULTS IN A MORE ACCURATE, BUT SPARSER REPRESENTATION OF THE PATCH!
	};
	enum LBPBINNING {
		LCROSSCOLOUR = 0,     //here only orientations and magnitudes are comoputed by every colour channel, but they are binned to the same HOG of the cell. 
		LBYCOLOUR = 1,     //meaning that every colour channel will get its of HOG feature, and the feature vector becomes 3 x as long (when using colours)
	};
	enum UNIFORMITY {
		LPURE = 0,
		LUNIFORM = 1,
	};

	struct LBPSettings {
		unsigned int cellSize;             // assumes square cell. Best to make it an clean divisor of patchsize
		unsigned int cellStride;           // the stride the cell window makes when iterating over the patch. (This may also be the cell size itself for a seperation into quadrants)
		unsigned int patchSize;          //the size of the patch       
		UNIFORMITY uniform;				 //unsigned int numberOfCells;         //is internally computed by virtue of cell size, stride, and blocksize.
		bool useColourPixel;               //use the gray pixels? or all color channels. If all channels, then feature is 3 times as large. Default is true
		LBPPadding padding;     
		int LBPSize;
		LBPBINNING binmethod;          
		int debugLevel;
	};


	class LBPDescriptor {
			//used to track in what indexes we bin the values.
		//vector<int> uniformFeatureShifts;
		LBPSettings settings;
		vector<int> uniformFeatureIndex;

	public:
		LBPDescriptor();
		void setSettings(LBPSettings s);
		bool debugOut, normalOut;
		Feature getLBP(Patch& patch);
	private:
      //unsigned int computeLBP(unsigned int X, unsigned int Y, Patch& patch);
	  unsigned int computeLBP(unsigned int X, unsigned int Y, Patch patch, LBPColour col);

	  void binLBP(unsigned int X, unsigned int Y, LBPColour col, vector<float>& cellLBPHistogram, Patch block);

	  vector<float> computeCellLBP(unsigned int X, unsigned int Y, Patch patch);
	  int lbpdiff(unsigned int centX, unsigned int centY, unsigned int X, unsigned int Y, Patch patch, LBPColour col);
	  bool isUniform(int lbp);
	  int uniformValue(int lbp);
	};

	

}


#endif