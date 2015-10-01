
#include <csvm/csvm_lbp_descriptor.h>
#include <bitset>
#include <iostream>

using namespace std;
using namespace csvm;

LBPDescriptor::LBPDescriptor() {
	
}


//This function implements basic Local Binary Patterns, and returns a feature vector.
vector< int > LBPDescriptor::getLBP(Patch patch, int channel) {

	int patchWidth = patch.getWidth();
	int patchHeight = patch.getHeight();
	const int scope = 1; //the neighbourhood size we consider. possibly later to be a custom argument
	
	std::bitset<(8 * scope)> pixelFeatures;
	vector<int> histogram(255, 0); //initialize a histogram to represent a whole patch
	cout << " patch width is: " << patchWidth;
	//for now 

	//iterate ofer the whole patch, with boundary setoff of 1 
	for (int x = 1;x < patchWidth-1;++x) { 
		for (int y = 1;y < patchHeight-1;++y) {
			//get the pixel intensity value of the central pixel we occupy ourselves with..
			int centroidPixelIntensity = patch.getGreyPixel(x, y);

			//in a neighbourhood around the centroid pixel: 
			for (int dx = 0;dx < 2*scope; ++dx) {
				for (int dy = 0;dy < 2*scope; ++dy) {
					//use the setoff relative to the centroid as index-based accessor: 
					//the sum of dx+dy will 
					int neighbourPixelX = dx - scope;
					int neighbourPixelY = dy - scope;
					pixelFeatures[(3*dx) + dy] = ( (centroidPixelIntensity > patch.getGreyPixel(x + neighbourPixelX, y + neighbourPixelY)) ? 0 : 1);
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



