#include <csvm/csvm_mlp_controller.h>
 
/* This class will control the multiple MLPs for mlp pooling
 * TODO: The normalization now happens inside of this class, because it is not really part of an mlp.
 * 		  It still needs testing if it actually works.
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

void MLPController::setSettings(MLPSettings s){
	std::cout << "set settings..." << std::endl;
	mlps.reserve(s.nSplitsForPooling * s.nSplitsForPooling);
	for(int i = 0; i < (s.nSplitsForPooling * s.nSplitsForPooling);i++){
		MLPerceptron mlp;
		mlp.setSettings(s); 
		mlps.push_back(mlp);
	}
} 
 
void MLPController::setMinAndMaxValueNorm(Feature inputFeature){
	double possibleMaxValue = *std::max_element(inputFeature.content.begin(), inputFeature.content.end());
	double possibleMinValue = *std::min_element(inputFeature.content.begin(), inputFeature.content.end()); 
		
	if(possibleMaxValue > maxValue)
		maxValue = possibleMaxValue;
			
	if(possibleMinValue < minValue)
		minValue = possibleMinValue;
}

vector<Feature>& MLPController::normalized(vector<Feature>& inputFeatures){
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
 
void MLPController::initMLPs(){
	createDataBySquares();
}

void MLPController::createDataBySquares(){
	//trainingSet and validationSet are created so call by reference is possible.
	//since both are large vector's call by reference is faster than call by value
	
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
	return normalized(validationSet);
}

vector<Feature>& MLPController::createRandomFeatureVector(vector<Feature>& trainingData){
    unsigned int nPatches = settings.scannerSettings.nRandomPatches;
    
    int sizeTrainingSet = (int)(dataset.getTrainSize()*(1-settings.mlpSettings.crossValidationSize));
    
	vector<Feature> testData;
	
	minValue = 10000;
	maxValue = 0;

	std::cout << "cre1ating random feature vector... "<< std::endl;
	
	for(size_t pIdx = 0; pIdx < nPatches; ++pIdx){	  
      Patch patch = imageScanner.getRandomPatch(dataset.getTrainImagePtr(rand() % sizeTrainingSet));
      Feature newFeat = featExtr.extract(patch);
      newFeat.setSquareId(calculateSquareOfPatch(patch));
      
      setMinAndMaxValueNorm(newFeat);
      trainingData.push_back(newFeat);   
   }
	return normalized(trainingData);	
}
void MLPController::trainMLP(MLPerceptron& mlp,vector<Feature>& trainingSet, vector<Feature>& validationSet){
    double crossvalidationSize = dataset.getTrainSize() * settings.mlpSettings.crossValidationSize;
    int noPatchesPerSquare = validationSet.size()/crossvalidationSize; 
    mlp.train(trainingSet,validationSet,noPatchesPerSquare);
}

void MLPController::trainMutipleMLPs(){
	initMLPs();
	for(unsigned int i=0;i<mlps.size();i++){ 		
		trainMLP(mlps[i],splitTrain[i],splitVal[i]);
		std::cout << "(classifier) trained mlp["<<i<<"]" << std::endl;
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
		
	vector<vector<Feature> > testFeatures = splitUpDataBySquare(normalized(dataFeatures));
	
	vector<double> oneSquare = vector<double>(settings.mlpSettings.nOutputUnits,0);      
  
	for(unsigned int i=0;i<mlps.size();i++){
		oneSquare = mlps[i].classifyPooling(testFeatures[i]);
		for(int j =0;j<settings.mlpSettings.nOutputUnits;j++)
			votingHistogramAllSquares[j] += oneSquare[j];
	}
	
	unsigned int mostVotedClass=0;
	double highestClassProb=0.0;
	for(int i=0;i<settings.mlpSettings.nOutputUnits; i++){
		if(votingHistogramAllSquares[i]>highestClassProb){
			highestClassProb = votingHistogramAllSquares[i];
			mostVotedClass = i;
		}
	}
 	return mostVotedClass;
}
