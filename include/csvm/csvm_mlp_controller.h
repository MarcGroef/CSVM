#ifndef MLP_CONTROLLER_H
#define MLP_CONTROLLER_H

#include <vector>
#include <cstdlib>
#include <ctime>

#include "csvm_settings.h"
#include "csvm_dataset.h"
#include "csvm_image_scanner.h"
#include "csvm_feature.h"
#include "csvm_feature_extractor.h"
#include "csvm_mlp.h"
#include "csvm_mlp_stacked.h"

using namespace std;
namespace csvm{
	
	class MLPController{
		private:
		int first; //variable for dropout a bit dirty
		int nMLPs;
		int nHiddenBottomLevel;
		
		int trainSize;
		int trainSizeBottomLevel;
		int trainSizeFirstLevel;
		
		int validationSize;
		
		int amountOfPatchesImage;
		
                vector<string> poolingTypes;
                
		vector<double> minValues;
		vector<double> maxValues;
                
		vector<vector<vector<double> > > deltas;
		vector<vector<vector<vector<double> > > > weights;
		
		vector<int> numPatchesPerSquare;
		
		ImageScanner imageScanner;
		FeatureExtractor featExtr;
		
		CSVMSettings settings;
		CSVMDataset* dataset;
		
		vector<vector<MLPerceptron> > mlps;
		
		void setMinAndMaxValueNorm(vector<Feature>& inputFeatures,int index);	
		vector<Feature>& normalizeInput(vector<Feature>& allInputFeatures,int index);
		void changeRange(vector<Feature>& data, double newMin, double newMax);
		
		void createDataBottomLevel(vector<vector<Feature> >& splitTrain, vector<vector<Feature> >& splitVal);
		void createDataFirstLevel(vector<Feature>& inputTrainFirstLevel, vector<Feature>& inputValFirstLevel);
		
		vector<Feature>& createCompletePictureSet(vector<Feature>& validationSet,int start, int end);
		vector<Feature>& createRandomFeatureVector(vector<Feature>& trainingData);
		
		vector<vector<Feature> > splitUpDataBySquare(vector<Feature>& trainingSet);
				
		void setFirstLevelData(vector<vector<Feature> >& splitDataBottom,vector<Feature>& dataFirstLevel, int sizeData);
		
		void trainMLP(MLPerceptron& mlp,vector<Feature>& trainingSet, vector<Feature>& validationSet, int numPatchesPerSquare);
		
		unsigned int mlpClassify(Image* im);
		
		int calculateSquareOfPatch(Patch patch);
		
		void dropOutTesting(vector<vector<vector<double> > >& newWeights);
		
		void exportFeatureSet(string filename, vector<Feature>& featureVector);
		void importFeatureSet(string filename, vector<Feature>& featureVector);
		void exportTrainedMLP(string filename);
		void importPreTrainedMLP(string filename);
		public:
		
		MLPController(FeatureExtractor* fe, ImageScanner* imScan, CSVMSettings* se, CSVMDataset* ds);
		
		void setSettings(MLPSettings s);

		void trainMutipleMLPs();
		unsigned int mlpMultipleClassify(Image* im);
	};
}
#endif
