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
		ImageScanner imageScanner;
		FeatureExtractor featExtr;
		
		CSVMSettings settings;
		CSVMDataset dataset;
		
		double minValue;
		double maxValue;
				
		vector<vector<Feature> > splitTrain;
		vector<vector<Feature> > splitVal;
		vector<MLPerceptron> mlps;
		
		vector<Feature>& normalized(vector<Feature>& inputFeatures);
		void setMinAndMaxValueNorm(Feature inputFeature);

		
		void createDataBySquares();
		int calculateSquareOfPatch(Patch patch);
		vector<Feature>& createValidationSet(vector<Feature>& validationSet);
		vector<Feature>& createRandomFeatureVector(vector<Feature>& trainingData);
		void trainMLP(MLPerceptron& mlp,vector<Feature>& trainingSet, vector<Feature>& validationSet);
		vector<vector<Feature> > splitUpDataBySquare(vector<Feature>& trainingSet);
		void initMLPs();
		
		public:
		
		MLPController();
		MLPController(FeatureExtractor* fe, ImageScanner* imScan, CSVMSettings* se, CSVMDataset* ds);
		
		void setSettings(MLPSettings s);

		void trainMutipleMLPs();
		unsigned int mlpMultipleClassify(Image* im);
	};
}
#endif
