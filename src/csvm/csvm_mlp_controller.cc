#include <csvm/csvm_mlp_controller.h>
#include <math.h>
//~ #include <iostream>
//~ #include <fstream>

/* This class will control the multiple MLPs for mlp pooling
 * 
 */
 
 using namespace std;
 using namespace csvm;
 
MLPController::MLPController(FeatureExtractor* fe, ImageScanner* imScan, CSVMSettings* se, CSVMDataset* ds){
	featExtr = *fe;
	imageScanner = *imScan;
	settings = *se;
	dataset = *ds;
}

MLPController::MLPController(){
	std::cout<<"lekker"<<std::endl;	
}

double MLPController::good_exp(double y){
	if(y < -10.0) return 0;
	if(y > 10.0) return 9999999.0;
	
	return exp(y);
}

void MLPController::setSettings(MLPSettings s){
	std::cout << "set settings..." << std::endl;
	mlps.reserve(s.nSplitsForPooling * s.nSplitsForPooling);
	weightingMLPs.reserve(s.nSplitsForPooling * s.nSplitsForPooling);
	outputMLP = vector<double>(s.nOutputUnits,0.0);
	for(int i = 0; i < (s.nSplitsForPooling * s.nSplitsForPooling);i++){
		MLPerceptron mlp;
		s.isWeightingMLP = false;
		mlp.setSettings(s); 
		mlps.push_back(mlp);
	}
	for(int i = 0; i < (s.nSplitsForPooling * s.nSplitsForPooling);i++){
		MLPerceptron mlp;
		s.isWeightingMLP = true;
		s.nOutputUnits = 1;
		mlp.setSettings(s); 
		weightingMLPs.push_back(mlp);
	}
} 
 
void MLPController::initMLPs(){
	createDataBySquares();
}

void MLPController::createDataBySquares(){
	vector<Feature> trainingSet;
 	vector<Feature> validationSet;

	splitTrain = splitUpDataBySquare(createRandomFeatureVector(trainingSet));
	splitVal   = splitUpDataBySquare(createValidationSet(validationSet));
	
	trainingSet.clear();
	validationSet.clear();
}

int MLPController::calculateSquareOfPatch(Patch patch){
	int splits = settings.mlpSettings.nSplitsForPooling;
	
	int imWidth  = settings.datasetSettings.imWidth;
	int imHeight = settings.datasetSettings.imHeight;
	
/*	if (imWidth/splits < patch.getWidth() || imHeight/splits < patch.getHeight()){
		std::cout << "(mlp_controller) ERROR: Patch size is too large for the pools. Please change either the patch size and/or the 'nSplitsForPoolig' in the settings file." << std::endl;
		exit (-1);
	}
*/
	
	int middlePatchX = patch.getX() + patch.getWidth() / 2;
	int middlePatchY = patch.getY() + patch.getHeight() / 2;
	
	int square = middlePatchX / (imWidth/splits) + (splits * (middlePatchY / (imHeight/splits)));
	return square;
}

vector<Feature>& MLPController::createValidationSet(vector<Feature>& validationSet){
	int amountOfImagesCrossVal = dataset.getTrainSize() * settings.mlpSettings.crossValidationSize;

	vector<Patch> patches;   
    
	for(int i = dataset.getTrainSize() - amountOfImagesCrossVal; i < dataset.getTrainSize();i++){
		Image* im = dataset.getTrainImagePtr(i);
		 
		//extract patches
		patches = imageScanner.scanImage(im);
      
		//extract features from all patches
		for(size_t patch = 0; patch < patches.size(); ++patch){
			Feature newFeat = featExtr.extract(patches[patch]);
			newFeat.setSquareId(calculateSquareOfPatch(patches[patch]));
			validationSet.push_back(newFeat);
		}
	}
	return validationSet;
}

vector<Feature>& MLPController::createRandomFeatureVector(vector<Feature>& trainingData){
    unsigned int nPatches = settings.scannerSettings.nRandomPatches;
    
    int sizeTrainingSet = (int)(dataset.getTrainSize()*(1-settings.mlpSettings.crossValidationSize));
    
	vector<Feature> testData;
	std::cout << "creating random feature vector... "<< std::endl;
	
	for(size_t pIdx = 0; pIdx < nPatches; ++pIdx){
	  //std::cout << "\r" << round((100/(double) nPatches)*(double) pIdx) << "% done";
	  
      Patch patch = imageScanner.getRandomPatch(dataset.getTrainImagePtr(rand() % sizeTrainingSet));
      //std::cout << "(mlp controller) getSquare: " << patch.getSquare() << std::endl;
      Feature newFeat = featExtr.extract(patch);
      newFeat.setSquareId(calculateSquareOfPatch(patch));
      trainingData.push_back(newFeat);   
   }
   std::cout << std::endl;
	return trainingData;	
}
void MLPController::trainMLP(MLPerceptron& mlp,vector<Feature>& trainingSet, vector<Feature>& validationSet){
    double crossvalidationSize = dataset.getTrainSize() * settings.mlpSettings.crossValidationSize;
    int noPatchesPerSquare = validationSet.size()/crossvalidationSize; 
    mlp.train(trainingSet,validationSet,noPatchesPerSquare);
}

void MLPController::trainMutipleMLPs(){
	initMLPs();
	for(unsigned int i=0;i<mlps.size();i++){ 	
		setMinAndMaxValueNorm(splitTrain[i]);
		splitTrain[i] = normalizeInput(splitTrain[i]);
		splitVal[i] = normalizeInput(splitVal[i]);	
		
		std::cout << "(classifier) training mlp["<<i<<"]..." << std::endl;
		trainMLP(mlps[i],splitTrain[i],splitVal[i]);
		weightingMLPs[i].setDesiredOutputsForWeighting(mlps[i].getDesiredOutputsForWeighting());
		std::cout << "(classifier) training MLP for patch weights["<<i<<"]..." << std::endl;
		trainMLP(weightingMLPs[i],splitTrain[i],splitVal[i]);
  }
  splitTrain.clear();
  splitVal.clear();
}

vector<vector<Feature> > MLPController::splitUpDataBySquare(vector<Feature>& trainingSet){
	vector<vector<Feature> > splitBySquares = vector<vector<Feature> >(settings.mlpSettings.nSplitsForPooling * settings.mlpSettings.nSplitsForPooling);
			
	for(unsigned int i = 0;i < trainingSet.size();i++){
		splitBySquares[trainingSet[i].getSquareId()].push_back(trainingSet[i]);	
	}
	return splitBySquares;
}

vector<Feature>& MLPController::normalizeInput(vector<Feature>& inputFeatures){
	if (maxValue == 1.0 && minValue == 0.0)
		return inputFeatures;
	if (maxValue - minValue != 0){
		//normalize all the inputs
		for(unsigned int i = 0; i < inputFeatures.size();i++){
			for(int j = 0; j < inputFeatures[i].size;j++)
				inputFeatures[i].content[j] = (inputFeatures[i].content[j] - minValue)/(maxValue - minValue);
		}
	}else{
		for(unsigned int i = 0; i<inputFeatures.size();i++){
			for(int j = 0; j < inputFeatures[i].size;j++)
				inputFeatures[i].content[j] = 0;
		}
	}
	return inputFeatures;		
}

void MLPController::setMinAndMaxValueNorm(vector<Feature>& inputFeatures){
	minValue = inputFeatures[0].content[0];
	maxValue = inputFeatures[0].content[0];

	//compute min and max of all the inputs	
	for(unsigned int i = 0; i < inputFeatures.size();i++){
		double possibleMaxValue = *std::max_element(inputFeatures[i].content.begin(), inputFeatures[i].content.end());
		double possibleMinValue = *std::min_element(inputFeatures[i].content.begin(), inputFeatures[i].content.end()); 
		
		if(possibleMaxValue > maxValue)
			maxValue = possibleMaxValue;
			
		if(possibleMinValue < minValue)
			minValue = possibleMinValue;
	}
}


//----NEEDS TO BE CHANGED-----

void MLPController::activationsToOutputProbabilities(){
	double sumOfActivations = 0.0;
	for(int i = 0; i< settings.mlpSettings.nOutputUnits; i++){
		outputMLP[i] = good_exp(outputMLP[i]);
		sumOfActivations += outputMLP[i];
	}
	for(int i = 0; i< settings.mlpSettings.nOutputUnits; i++)
		outputMLP[i] /= sumOfActivations;
}


vector<double> MLPController::voting(vector<double> votingHistogram){
	activationsToOutputProbabilities();
	if(settings.mlpSettings.voting == "MAJORITY")
		return majorityVoting(votingHistogram);
	else if (settings.mlpSettings.voting == "SUM")
		return sumVoting(votingHistogram);
	else{ 
		std::cout << "This voting type is unknown. Change to a known voting type in the settings file" << std::endl; exit(-1);
	}
		
}

vector<double> MLPController::majorityVoting(vector<double> votingHistogram){       // get rid of votingHistogram[]
	int indexHighestAct = 0;
	double highestActivationClass = 0;
	
	for (int i=0; i<settings.mlpSettings.nOutputUnits;i++){
		if(outputMLP[i]>highestActivationClass){
			highestActivationClass = outputMLP[i];
			indexHighestAct = i;
		}	
	}
	votingHistogram[indexHighestAct] += 1;
	return votingHistogram;
}

vector<double> MLPController::sumVoting(vector<double> votingHistogram){
	for (int i=0; i<settings.mlpSettings.nOutputUnits;i++)
			votingHistogram[i] += outputMLP[i];	
	return votingHistogram;
}

unsigned int MLPController::mostVotedClass(vector<double> votingHistogram){
	unsigned int mostVotedClass = 0;
	double voteCounter = 0;
	
	for (int i = 0; i < settings.mlpSettings.nOutputUnits; i++){
		if (votingHistogram[i] > voteCounter){   //what happens if two classes have the same amount of votes? the later is chos(higher label)
			voteCounter = votingHistogram[i];
			mostVotedClass = i;
		}
	}
	return mostVotedClass;
}
//returns the summed output for one square
vector<double> MLPController::classifyImageSquare(MLPerceptron firstMLP, MLPerceptron weightingMLP, vector<Feature> features){
	vector<double> summedOutput (settings.mlpSettings.nOutputUnits,0.0);
	vector<double> weight (1,0.0);
	
	//ofstream myfile;
	//myfile.open("weightsVSoutputprobabilities.txt");
	//myfile << "weights \t outputprobbilities" << std::endl;
	
	for(unsigned int i=0;i<features.size();i++){
		outputMLP = firstMLP.runFeatureThroughMLP(features[i]);
		weight = weightingMLP.runFeatureThroughMLP(features[i]);
		//myfile << weight[0] << " \t " << outputMLP[features[i].getLabelId()] << std::endl;
		for(int j=0;j<settings.mlpSettings.nOutputUnits;j++){
			outputMLP[j] *= weight[0];					//weighing the outputs
		}
		summedOutput = voting(summedOutput);
	}
	//myfile.close();
	return summedOutput;
}

unsigned int MLPController::mlpMultipleClassify(Image* im){
  vector<Patch> patches;
	vector<Feature> dataFeatures;
	vector<double> votingHistogramAllSquares = vector<double>(settings.mlpSettings.nOutputUnits,0);      

	//extract patches
  patches = imageScanner.scanImage(im);

  //allocate for new features
  dataFeatures.reserve(patches.size());
      
  //extract features from all patches
  for(size_t patch = 0; patch < patches.size(); ++patch){
		Feature newFeat = featExtr.extract(patches[patch]);
		newFeat.setSquareId(calculateSquareOfPatch(patches[patch]));
		dataFeatures.push_back(newFeat);
	}
		
	vector<vector<Feature> > testFeatures = splitUpDataBySquare(dataFeatures);
	vector<double> oneSquare = vector<double>(settings.mlpSettings.nOutputUnits,0);      
  
  for(unsigned int i=0;i<mlps.size();i++){
		testFeatures[i] = normalizeInput(testFeatures[i]);
		oneSquare = classifyImageSquare(mlps[i],weightingMLPs[i],testFeatures[i]);
		
		for(int j =0;j<settings.mlpSettings.nOutputUnits;j++)
			votingHistogramAllSquares[j] += oneSquare[j];
	}
	

 	return mostVotedClass(votingHistogramAllSquares);
}
//-----/NEEDS TO BE CHANGED-----
