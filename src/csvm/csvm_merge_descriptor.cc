#include <csvm/csvm_merge_descriptor.h>
#include <iomanip> //for setprecision couting
#include <limits>

/*Requires more eleborate comments from Jonathan */

using namespace std;
using namespace csvm;


//HOGDescriptor::HOGDescriptor(int nBins = 9, int cellSize = 3, int blockSize = 9, bool useGreyPixel = 1) {
MERGEDescriptor::MERGEDescriptor() {

}


void MERGEDescriptor::setSettings(MERGESettings s){
	if(debugOut)cout << "mergesettings set" << endl;
   settings = s;
}


void MERGEDescriptor::setHOGSettings(HOGSettings hs) {
	if(debugOut)cout << "hog stuff set" << endl;
	settings.hogSettings = hs;
	//hog.setSettings(settings.hogSettings);
}

Feature MERGEDescriptor::normalizeFeature(Feature feat) {
    size_t featureLen = feat.size;
	float vTwoSquared = 0.0;

	// */
	//		L2 NORMALIZATION
	for (size_t idx = 0; idx < featureLen; ++idx) {
		vTwoSquared += pow(feat.content[idx], 2);
	}
	//   vTwoSquared = sqrt(vTwoSquared); //is now vector length

	// e is some magic number still...
	float e = 0.000000000000001;
	for (size_t idx = 0; idx < featureLen; ++idx) {
		feat.content[idx] /= sqrt(vTwoSquared + pow(e, 2));
	}
	// */
	//	CONVENTIONAL NORMALIZATION
	/*
	float max = 0.0;
	float min = numeric_limits<float>::max();
	for (size_t idx = 0; idx < featureLen; ++idx) {
		max = feat.content[idx] > max ? feat.content[idx]  : max;
		min = feat.content[idx] < min ? feat.content[idx] : min;
	}
	//updating content with normalized values:
	float diff = max - min;
	for (size_t idx = 0; idx < featureLen; ++idx) {
		feat.content[idx] = (feat.content[idx] - min) / (diff); 
	}

	*/



	return feat;
}




Feature MERGEDescriptor::standardizeFeature(Feature feat) {
	float mean = 0.0;
	float standardDeviation = 0.0;
	size_t featureLen = feat.size;

	for (size_t idx = 0; idx < featureLen; ++idx) {
		mean += feat.content[idx];
	}
	mean /= (float)featureLen;
	float deviation = 0.0;
	for (size_t idx = 0; idx < featureLen; ++idx) {
		deviation += pow((feat.content[idx] - mean), 2.0);
	}
	deviation /= (float)featureLen;
	standardDeviation = sqrt(deviation);

	for (size_t idx = 0; idx < featureLen; ++idx) {
		feat.content[idx] = (feat.content[idx] - mean) / standardDeviation;
	}

	return feat;
}

Feature MERGEDescriptor::getMERGE(Patch& p, CleanDescriptor& pix, HOGDescriptor& hog){
	

	settings.patchSize = p.getHeight();
	//cout << "getting Merge" << endl;
	
	Feature rawPixs = pix.describe(p);
	Feature hogFeature = hog.getHOG(p);

	size_t pixLen = rawPixs.size;
	size_t hogLen = hogFeature.size;

	size_t totalFeatureLength = pixLen + hogLen;
	
	vector <float> results(totalFeatureLength, 0.0);
	for (size_t idx = 0; idx < pixLen; ++idx) {
		results[idx] = (rawPixs.content[idx] * settings.weightRatio) / pixLen;
	}

	for (size_t idx = 0; idx < hogLen; ++idx) {
		results[pixLen + idx] = (hogFeature.content[idx] * (1 - settings.weightRatio)) / hogLen;
	}

	Feature result(results);
	result.label = p.getLabel();
	result.labelId = p.getLabelId();

   return result;
   
}

