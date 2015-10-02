
#include <csvm/csvm_lbp_descriptor.h>


using namespace std;
using namespace csvm;

LBPDescriptor::LBPDescriptor() {
	
}


//This function implements basic Local Binary Patterns, and returns a feature vector.
Feature LBPDescriptor::getLBP(Patch patch, int channel) {

	int patchWidth = patch.getWidth();
	int patchHeight = patch.getHeight();
	//const int scope = 1; //the neighbourhood size we consider. possibly later to be a custom argument
	
	bitset<(8)> pixelFeatures;
   Feature histogram(256,0);
   histogram.label = patch.getLabel();
	//vector<int> histogram(255, 0); //initialize a histogram to represent a whole patch
	//cout << " patch width is: " << patchWidth;
	//for now 

	//iterate ofer the whole patch, with boundary setoff of 1 
	for (int x = 1;x < patchWidth-1;++x) { 
		for (int y = 1;y < patchHeight-1;++y) {
			//get the pixel intensity value of the central pixel we occupy ourselves with..
			int centroidPixelIntensity = patch.getGreyPixel(x, y);

			//in a neighbourhood around the centroid pixel: 
			pixelFeatures[0] = ((centroidPixelIntensity > patch.getGreyPixel(x - 1, y - 1)) ? 0 : 1);
			pixelFeatures[1] = ((centroidPixelIntensity > patch.getGreyPixel(x, y - 1)) ? 0 : 1);
			pixelFeatures[2] = ((centroidPixelIntensity > patch.getGreyPixel(x + 1, y - 1)) ? 0 : 1);
			pixelFeatures[3] = ((centroidPixelIntensity > patch.getGreyPixel(x + 1, y)) ? 0 : 1);
			pixelFeatures[4] = ((centroidPixelIntensity > patch.getGreyPixel(x + 1, y + 1)) ? 0 : 1);
			pixelFeatures[5] = ((centroidPixelIntensity > patch.getGreyPixel(x, y + 1)) ? 0 : 1);
			pixelFeatures[6] = ((centroidPixelIntensity > patch.getGreyPixel(x - 1, y + 1)) ? 0 : 1);
			pixelFeatures[7] = ((centroidPixelIntensity > patch.getGreyPixel(x - 1, y)) ? 0 : 1);
			


			/*
			for (int dx = 0;dx < 2*scope; ++dx) {
				for (int dy = 0;dy < 2*scope; ++dy) {
					//use the setoff relative to the centroid as index-based accessor: 
					//the sum of dx+dy will 
					int neighbourPixelX = dx - scope;
					int neighbourPixelY = dy - scope;
					//if (not (neighbourPixelX == 0 && neighbourPixelY == 0)) 
					pixelFeatures[((3*dx) + dy)] = ( (centroidPixelIntensity > patch.getGreyPixel(x + neighbourPixelX, y + neighbourPixelY)) ? 0 : 1);
				}
			}
			*/
			//transpose pixelfeatures to byte value

			//cout << "test";
			//cout << pixelFeatures.to_ulong() << "\n";

			histogram.content[(int)pixelFeatures.to_ulong()] += 1;

		}
	}

	return histogram;
}



