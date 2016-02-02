
#ifndef CSVM_MERGE_DESCRIPTOR_H
#define CSVM_MERGE_DESCRIPTOR_H

/*Requires more eleborate comments from Jonathan */

#include <vector>
#include <iostream>
#include <cmath>
#include "csvm_patch.h"
#include "csvm_image.h"
#include "csvm_feature.h"
#include "csvm_clean_descriptor.h"
#include "csvm_hog_descriptor.h"

using namespace std;


namespace csvm {
	
/*	enum Colour {
		GRAY = -1,
		RED = 0,
		GREEN = 1,
		BLUE = 2,
	};

	enum Padding {
		ZERO = 0,                  //the type of padding used
		IDENTITY = 1,
		NONE = 2,
	};

	enum INTERPOLATION {
		INTERPOLATE_BINARY = 0,
		INTERPOLATE_LINEAR = 1,
		INTERPOLATE_TRILINEAR = 2,
	};

	enum HOGGRADIENT {
		MAGNITUDE = 0,
		ORIENTATION = 1,
	};
	
	*/


	struct MERGESettings {
		HOGSettings hogSettings;
		int patchSize;
		bool useGreyPixel;
		double weightRatio;
	};
	class MERGEDescriptor {
		MERGESettings settings;
		//HOGDescriptor hog;
		//CleanDescriptor clean;
	public:
		MERGEDescriptor();
		//MERGEDescriptor(int cellSize, int cellStride, int blockSize);
		void setSettings(MERGESettings s);
		void setHOGSettings(HOGSettings hs);
		//HOGDescriptor(int nBins, int cellSize, int blockSize, bool useGreyPixel); 
		//HOGDescriptor(int nBins, int numberOfCells, int blockSize, bool useGreyPixel);
		Feature getMERGE(Patch& p, CleanDescriptor& pix, HOGDescriptor& hog);

	private:
		Feature standardizeFeature(Feature feat);
		Feature normalizeFeature(Feature feat);
		//Feature weightFeature(Feature feat);
		//Feature standardizeFeature(vector<double> content);
		//Feature normalizeFeature(vector<double> content);
	
	};

}


#endif
