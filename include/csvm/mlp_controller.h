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
		
		
		
		vector<MLPerceptron> mlps;
		
		ImageScanner imageScanner;
		FeatureExtractor featExtr;
		
		CSVMSettings settings;
		CSVMDataset dataset;
		
		
		public:
		void initMLPs();
		vector<Feature>& createValidationSet(vector<Feature>& validationSet);
		vector<Feature>& createRandomFeatureVector(vector<Feature>& trainingData);
		void trainMLP(MLPerceptron& mlp,vector<Feature>& trainingSet, vector<Feature>& validationSet);
		void trainMutipleMLPs();
		vector<vector<Feature> > splitUpDataBySquare(vector<Feature>& trainingSet);
		unsigned int mlpClassify(Image* im);
		unsigned int mlpMultipleClassify(Image* im);
		
		MLPController();
		MLPController(FeatureExtractor* fe, ImageScanner* imScan, CSVMSettings* se, CSVMDataset* ds);

		
	};
}
#endif
