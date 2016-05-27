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


using namespace std;
namespace csvm{
	
	class MLPController{
		private:
		int nMLPs;
		int validationSize;
		int trainSize;
		double minValue;
		double maxValue;
      
		vector<int> numPatchesPerSquare;
		
		ImageScanner imageScanner;
		FeatureExtractor featExtr;
		
		CSVMSettings settings;
		CSVMDataset dataset;
				
		vector<vector<Feature> > splitTrain;
		vector<vector<Feature> > splitVal;
		
		vector<Feature> inputTrainSecondLayerMLP;
		vector<Feature> inputValSecondLayerMLP;
		
		vector<vector<MLPerceptron> > mlps;
		
		void createDataFirstLayerMLP();
		void createDataSecondLayerMLP();
	
		vector<Feature>& createCompletePictureSet(vector<Feature>& validationSet,int start, int end);
		vector<Feature>& createRandomFeatureVector(vector<Feature>& trainingData);
		vector<vector<Feature> > splitUpDataBySquare(vector<Feature>& trainingSet);
		
		void setMinAndMaxValueNorm(vector<Feature>& inputFeatures);	
		vector<Feature>& normalizeInput(vector<Feature>& allInputFeatures);
		
		void createOutputProbabilitiesVectorTest(vector<vector<Feature> >& testSet);
		void createOutputProbabilitiesVectorTrain(vector<vector<Feature> >& trainingSet, vector<vector<Feature> >& validationSet);
		
		unsigned int mlpClassify(Image* im);
		int calculateSquareOfPatch(Patch patch);
		
		void trainMLP(MLPerceptron& mlp,vector<Feature>& trainingSet, vector<Feature>& validationSet);

	    
		
		public:
		
		MLPController(FeatureExtractor* fe, ImageScanner* imScan, CSVMSettings* se, CSVMDataset* ds);
		
		void setSettings(MLPSettings s);

		void trainMutipleMLPs();
		unsigned int mlpMultipleClassify(Image* im);
	};
}
#endif
