
#include <csvm/csvm_lbp_descriptor.h>


using namespace std;
using namespace csvm;

LBPDescriptor::LBPDescriptor() {
	//cout << "we reached 1!";
	int findex = 1;
	featureIndex = vector<int>(256,0);
	for (int idx = 0; idx < 256; ++idx) {
		//if a number is uniform, then we want to record it. 
		if (isUniform(idx)) {
			featureIndex[idx] = findex;
			++findex;
		}
		else {
			//if it is not uniform, we refer to index 0, where we put in all non-uniform lbps
			featureIndex[idx] = 0;
		}
	}
	//cout << "we reached 2!";
}


//this function is used to check whether a certain value is uniform or not. 
bool LBPDescriptor::isUniform(int lbp) {
	//cout << "isUniform called";
	int numberOfTrans = 0;
	bitset<8> lbpbits(lbp);
	for (int idx = 0; idx < 7; ++idx) {
		if (lbpbits[idx] != lbpbits[idx + 1]){
			++numberOfTrans;
		}
	}

	return (numberOfTrans <= 2);
}



//is used to check the uniform equivalent of a function. 
int LBPDescriptor::uniformValue(int lbp) {
	//cout << "uniformvalue called with " << lbp << '\n';
	//iterate down a byte
	int uniformval = 0;
	int lbpvalue = lbp;
	for (int bit = 7; bit > 0; --bit) {
		//if we are at a 1-bit
		if (lbpvalue - pow(2, bit) >= 0) {
			lbpvalue -= pow(2, bit);
			uniformval += (bit*(bit-1)) + 2;
		}
	}
	//cout << "uniformvalue returned " << uniformval << '\n';
	return uniformval;
}



//This function implements basic Local Binary Patterns, and returns a feature vector.
Feature LBPDescriptor::getLBP(Patch patch, int channel) {
	//cout << "getLBP called" << '\n';
	int patchWidth = patch.getWidth();
	int patchHeight = patch.getHeight();
	//const int scope = 1; //the neighbourhood size we consider. possibly later to be a custom argument
	
	bitset<(8)> pixelFeatures;
	int biggestvalue = 0;
	for (int idx = 0; idx < featureIndex.size(); ++idx) {
		biggestvalue = (featureIndex[idx] > biggestvalue ? featureIndex[idx] : biggestvalue);
	}
   Feature histogram( biggestvalue +1 ,0);
   histogram.label = patch.getLabel();
	//vector<int> histogram(255, 0); //initialize a histogram to represent a whole patch
	//cout << " patch width is: " << patchWidth;
	//for now 

	//iterate ofer the whole patch, with boundary setoff of 1 
   //cout << "we reached 1!" << '\n';

   //for (size_t idx = 0; idx < featureIndex.size(); ++idx) {
//	   cout << "< " << idx << " , " << featureIndex[idx] << ">	||	 ";
 //  }
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

			histogram.content[ featureIndex[(int)pixelFeatures.to_ulong()]] += 1;

		}
	}
	//cout << "we reached 2!" << '\n';
	/*for (size_t idx = 0; idx < histogram.content.size(); ++idx) {
		cout << "element " << idx << " = " << histogram.content[idx] << endl;
	}*/
	//cout << '\n';
	for(size_t b = 0; b < histogram.content.size(); ++b)
      histogram.content[b] /= ( (patchWidth - 2) * (patchWidth - 2));
	return histogram;
}



