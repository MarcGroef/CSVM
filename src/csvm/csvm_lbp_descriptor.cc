
#include <csvm/csvm_lbp_descriptor.h>
#include <io.h>
#include <bitset>
#include <iostream>

using namespace std;
using namespace csvm;



LBPDescriptor::LBPDescriptor() {
	
}

//assumes a square uchar (grey) image


//This function implements classic HOG, including how to partitionize the image. CSVM will do this in another way, so it's not quite finished
vector< int > LBPDescriptor::getLBP(Patch patch, int channel) {

	int patchWidth = patch.getWidth();
	int patchHeight = patch.getHeight();
	const int scope = 1; //the neighbourhood size we consider. possibly later to be a custom argument
	
	std::bitset<(4 * scope)> pixelFeatures;
	vector<int> histogram(255, 0); //initialize a histogram to represent a whole patch
	cout << " patch width is: " << patchWidth;
	//for now 

	for (int x = 1;x < patchWidth-1;++x) { // iterating over the x axis, up to the the boundary-1. we want to perform computations.
		for (int y = 1;y < patchHeight-1;++y) {
			int centroidPixelIntensity = patch.getGreyPixel(x, y);

			for (int dx = 0;dx < 2*scope; ++dx) {
				for (int dy = 0;dy < 2*scope; ++dy) {
					int neighbourPixelX = dx - scope;
					int neighbourPixelY = dy - scope;
					pixelFeatures[dx + dy] = ( (centroidPixelIntensity > patch.getGreyPixel(x + neighbourPixelX, y + neighbourPixelY)) ? 0 : 1);
				}
			}
			//transpose pixelfeatures to byte value
			cout << "test";
			cout << histogram[pixelFeatures.to_ulong()] << "	";
			histogram[pixelFeatures.to_ulong()] += 1;

		}
	}

	return histogram;
}



