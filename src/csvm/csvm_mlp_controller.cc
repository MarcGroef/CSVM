#include <csvm/csvm_mlp_controller.h>
#include <math.h>
#include <sys/stat.h>
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
	if(settings.mlpSettings.useWeightingMLPs){
		for(int i = 0; i < (s.nSplitsForPooling * s.nSplitsForPooling);i++){
			MLPerceptron mlp;
			s.isWeightingMLP = true;
			s.nOutputUnits 				= s.weightingOutputUnits;
			s.nInputUnits  				= s.weightingInputUnits;
			s.nHiddenUnits				= s.weightingHiddenUnits;
			s.nLayers					= s.weightingLayers;
      
			s.learningRate				= s.weightingLearningRate;
			s.voting					= s.weightingVoting;
			s.trainingType				= s.weightingTrainingType;
			s.crossValidationInterval 	= s.weightingCrossValidationInterval;
			s.crossValidationSize		= s.weightingCrossValidationSize;
			s.epochs					= s.weightingEpochs;
			s.stoppingCriterion			= s.weightingStoppingCriterion;
			
			mlp.setSettings(s); 
			weightingMLPs.push_back(mlp);
		}
	}
	amountOfPatchesImage = (settings.datasetSettings.imWidth - settings.scannerSettings.patchWidth + 1) * (settings.datasetSettings.imHeight - settings.scannerSettings.patchHeight + 1);
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
   setMinAndMaxValueNorm(trainingData);
   trainingData = normalizeInput(trainingData);
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
		std::cout << "(classifier) training mlp["<<i<<"]..." << std::endl;
		trainMLP(mlps[i],splitTrain[i],splitVal[i]);
		if(settings.mlpSettings.useWeightingMLPs){
			weightingMLPs[i].setDesiredOutputsForWeighting(mlps[i].getDesiredOutputsForWeighting());
			std::cout << "(classifier) training MLP for patch weights["<<i<<"]..." << std::endl;
			trainMLP(weightingMLPs[i],splitTrain[i],splitVal[i]);
			cout << "\nHistogram of output probabilities: \n";
			printHistogram(mlps[i].getDesiredOutputsForWeighting(), 20);
		}
  }
  splitTrain.clear();
  splitVal.clear();
}

void MLPController::printHistogram(vector<double> data, int bins){
	vector<int> histogram (bins,0);
	double max = 0.0;
	
	
	for(unsigned int i=0; i<data.size() ; i++){
		histogram[(int)(data[i]*bins/2)] ++;
		if(data[i] > max)
			max = data[i];
	}
	for(int i=0; i<bins; i++)
		cout << histogram[i] << "\t|";
	std::cout << endl;
	for(int i=0; i<bins; i++)
		cout << "--------";
	cout << "-" << endl;
	for(float i=0; i<bins; i++)
		std::cout << setprecision (3) << fixed << (i+1)/(bins/2) << "\t|";
	cout << "\n";
	cout << "max value: "  << max << endl;
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

vector<double> MLPController::majorityVoting(vector<double> votingHistogram){ 
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
vector<double> MLPController::classifyImageSquare(int indexOfMLPs, vector<Feature> features){
	vector<double> summedOutput (settings.mlpSettings.nOutputUnits,0.0);
	vector<double> weight (1,0.0);
	MLPerceptron firstMLP = mlps[indexOfMLPs];
	MLPerceptron weightingMLP;
	if(settings.mlpSettings.useWeightingMLPs){
		weightingMLP = weightingMLPs[indexOfMLPs];
	}
	
	//ofstream myfile;
	//myfile.open("weightsVSoutputprobabilities.txt");
	//myfile << "weights \t outputprobbilities" << std::endl;
	
	for(unsigned int i=0;i<features.size();i++){
		outputMLP = firstMLP.runFeatureThroughMLP(features[i]);
		if(settings.mlpSettings.useWeightingMLPs)
			weight = weightingMLP.runFeatureThroughMLP(features[i]);
		else 
			weight[0] =1.0;
			//myfile << weight[0] << " \t " << outputMLP[features[i].getLabelId()] << std::endl;
		allWeights.push_back(weight[0]);
		for(int j=0;j<settings.mlpSettings.nOutputUnits;j++){
			outputMLP[j] *= weight[0];					//weighing the outputs
		}
		
		//outputMLP is global and used in the voting functions
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
		oneSquare = classifyImageSquare(i,testFeatures[i]);
		
		for(int j =0;j<settings.mlpSettings.nOutputUnits;j++)
			votingHistogramAllSquares[j] += oneSquare[j];
	}
	counter ++;
	if(counter%settings.datasetSettings.nTestImages == 0){
		cout << "\nHistogram weights\n";
		printHistogram(allWeights, 20);
	}
 	return mostVotedClass(votingHistogramAllSquares);
}

//---------------------start:import/export---------------------------
union charDouble{
   char chars[8];
   double doubleVal;
};

union charInt{
   char chars[4];
   unsigned int intVal;
};

bool fileExists(const std::string& filename)
{
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1)
    {
        return true;
    }
    return false;
}


void MLPController::exportFeatureSet(string filename, vector<Feature>& featureVector){
	//TODO: Also export training and testing images pointers. Otherwise it goes wrong in the testing phase, because now you have no idea on which images you trained and on which you did not.
 
   /* Featureset file conventions:
    * 
    * first,  Dataset			: 0, CIFAR10 1,MNIST   			(4 bytes)
    * second, Amount of features: 0-10.000.000         			(4 bytes)
    * third,  PatchWidth		: 0-36                 			(4 bytes)
    * fourth, PatchHeigth		: 0-36                 			(4 bytes)  
    * fifth,  FeatSize			: 0-10.000             		   	(4 bytes)
    * sixth,  FeatExtractor		: 0,LBP 1,CLEAN 2,HOG 3,MERGE   (4 bytes)  
    * 
    * from now one it will look like this:
    * 	all double values from the feature
    * 	labelId of the feature
    * 	pool it came from
    *  
    *  No seperator characters are used
   */
   
   charInt fancyInt;
   charDouble fancyDouble;
 
   ofstream file(filename.c_str(),  ios::binary);
   
   //write dataset used
	switch(settings.datasetSettings.type){
      case DATASET_CIFAR10:
         fancyInt.intVal=0;
         break;
      case DATASET_MNIST:
		 fancyInt.intVal=1;
         break;   
	}
   file.write(fancyInt.chars, 4);
 
   //write amount of features
   fancyInt.intVal = settings.scannerSettings.nRandomPatches;   
   file.write(fancyInt.chars, 4);
   
   //write patchWidth
   fancyInt.intVal = settings.scannerSettings.patchWidth;
   file.write(fancyInt.chars, 4);
   
   //write patchWidth
   fancyInt.intVal = settings.scannerSettings.patchHeight;
   file.write(fancyInt.chars, 4);
   
   //write FeatSize
   fancyInt.intVal = settings.mlpSettings.nInputUnits;
   file.write(fancyInt.chars, 4);   
   
   //write FeatExtractor method
   	switch(settings.featureSettings.featureType){
      case LBP:
         fancyInt.intVal=0;
         break;
      case CLEAN:
		 fancyInt.intVal=1;
         break;
      case HOG:
		 fancyInt.intVal=2;
         break;
      case MERGE:
		 fancyInt.intVal=3;
         break;
	}
   file.write(fancyInt.chars, 4); 

	for(unsigned int i=0;i<featureVector.size();i++){
		for(int j=0;j<featureVector[i].size;j++){
			fancyDouble.doubleVal = featureVector[i].content[j];
			file.write(fancyDouble.chars, 8);
		}
		//write label of the feat
		fancyInt.intVal = featureVector[i].getLabelId();
		//std::cout << "labels written to the file: " << featureVector[i].getLabelId() << std::endl
		file.write(fancyInt.chars, 4);
		
		//write pool it came from
		fancyInt.intVal = featureVector[i].getSquareId(); //does this always exists??
		file.write(fancyInt.chars, 4);		
	}		
	if(featureVector.size() == settings.scannerSettings.nRandomPatches){
		//write min value
		fancyDouble.doubleVal = minValue; 
		file.write(fancyDouble.chars, 8);
		//std::cout << "min value in write: " << fancyDouble.doubleVal << std::endl;
		//write max value
		fancyDouble.doubleVal = maxValue;
		file.write(fancyDouble.chars, 8);
		//std::cout << "min value in write: " << fancyDouble.doubleVal << std::endl;

		vector<unsigned int> trainImages = dataset.getTrainImageNums();
		for(int i=0;i<dataset.getTrainSize();i++){
			fancyInt.intVal = trainImages[i];
			//std::cout << trainImages[i] << std::endl;
			file.write(fancyInt.chars,4);
		}
		vector<unsigned int> testImages = dataset.getTestImageNums();
		for(int i=0;i<dataset.getTestSize();i++){
			fancyInt.intVal = testImages[i];
			file.write(fancyInt.chars,4);
		}
	}
   file.close();
}

void MLPController::importFeatureSet(string filename, vector<Feature>& featureVector){
	charInt fancyInt;
	charDouble fancyDouble;
  
	//unsigned int typesize;
	//unsigned int featDims;
	
	ifstream file(filename.c_str(), ios::binary);
   
 	if(!fileExists(filename.c_str())){
		std::cout << "The filename: " << filename.c_str() << "cannot be found, please change this in the setting file" << std::endl;
		exit(-1);		
	}
   
	file.read(fancyInt.chars,4);
	unsigned int datasetNum = fancyInt.intVal; //some check that his num is smaller than 2
	CSVMDatasetType readInDatasetType;
   
   	switch(datasetNum){
      case 0:
         readInDatasetType = DATASET_CIFAR10;
         break;
      case 1:
		 readInDatasetType = DATASET_MNIST;
		 break;
	   default:
		readInDatasetType = DATASET_CIFAR10;
		std::cout << "The read in dataset is unknown, by default it is set to CIFAR10" << std::endl;

	}
   
	if(readInDatasetType != dataset.getType()){
		std::cout << "The dataset that is read in is " << readInDatasetType << " and in the settings file you have " << dataset.getType() << ", please change this" << std::endl;
		exit(-1);
	}
	
	file.read(fancyInt.chars,4);
	unsigned int readInNRandomPatches = fancyInt.intVal;
	if(settings.scannerSettings.nRandomPatches != readInNRandomPatches){
		std::cout << "The nRandomPatches that is read in is " << readInNRandomPatches << " in the settings file it is " << settings.scannerSettings.nRandomPatches << ", please change this" << std::endl;
		exit(-1);
	}
  
	file.read(fancyInt.chars,4);
	unsigned int readInPatchWidth = fancyInt.intVal;
	if(settings.scannerSettings.patchWidth != readInPatchWidth){
		std::cout << "The patchWidth that is read in is " << readInPatchWidth << " in the settings file it is " << settings.scannerSettings.patchWidth << ", please change this" << std::endl;
		exit(-1);
	}
  
	file.read(fancyInt.chars,4);
	unsigned int readInPatchHeigth = fancyInt.intVal;
	if(settings.scannerSettings.patchHeight != readInPatchHeigth){
		std::cout << "The patchHeigth that is read in is " << readInPatchHeigth << " in the settings file it is " << settings.scannerSettings.patchHeight << ", please change this" << std::endl;
		exit(-1);
	}

	file.read(fancyInt.chars,4);
	int readInFeatSize = fancyInt.intVal;
	if(settings.mlpSettings.nInputUnits != readInFeatSize){
		std::cout << "The feature size that is read in is " << readInFeatSize << " in the settings file it is " << settings.mlpSettings.nInputUnits << ", please change this" << std::endl;
		exit(-1);
	}
	file.read(fancyInt.chars,4);
	unsigned int FeatExtNum = fancyInt.intVal; //Somecheck that this is a num smaller than 4.
	FeatureType readInFeatExt;
	
	switch(FeatExtNum){
      case 0:
         readInFeatExt=LBP;
         break;
      case 1:
		 readInFeatExt=CLEAN;
         break;
      case 2:
		 readInFeatExt=HOG;
         break;
      case 3:
		 readInFeatExt=MERGE;
	  default:
		 readInFeatExt=HOG;
		 std::cout << "The read in feature extractor is unknown, by default it is set to HOG" << std::endl;
	}
	
	if(settings.featureSettings.featureType != readInFeatExt){
		std::cout << "The Feature extractor that is read in is " << readInFeatExt << " in the settings file it is" << settings.featureSettings.featureType << ", please change this" << std::endl;
		exit(-1);		
	}
	int sizeOfFeatureVector;
	
	if(filename.find("RandomFeat") == 0){
		sizeOfFeatureVector = readInNRandomPatches;
	}
	else {
		sizeOfFeatureVector = dataset.getTrainSize() * settings.mlpSettings.crossValidationSize * amountOfPatchesImage;
	}

	for(int i=0;i<sizeOfFeatureVector;i++){
		vector<double> contentFeat;
		contentFeat.reserve(readInFeatExt);
		
		for(int j=0;j<readInFeatSize;j++){
			file.read(fancyDouble.chars, 8);
			contentFeat.push_back(fancyDouble.doubleVal);
		}
		Feature feat = new Feature(contentFeat);
		
		//read label id
		file.read(fancyInt.chars, 4);
		feat.setLabelId(fancyInt.intVal);
		
		//read square id
		file.read(fancyInt.chars, 4);
		feat.setSquareId(fancyInt.intVal);
		
		featureVector.push_back(feat);
	}
	
	//Check if the train/test images that are read in are the same as in the settings file
	//This only needs to happen with the training set.. not the validation set
	
	if(!(file.peek() == std::ifstream::traits_type::eof())){
		//read min value
		file.read(fancyDouble.chars, 8);
		minValue = fancyDouble.doubleVal;
	
		//read max value
		file.read(fancyDouble.chars, 8);
		maxValue = fancyDouble.doubleVal;
		
		vector<unsigned int> readInTrainImages;
		for(int i=0;i<dataset.getTrainSize();i++){
			file.read(fancyInt.chars,4);
			//std::cout << "imageNum["<<i<<"]: " << fancyInt.intVal << std::endl;
			readInTrainImages.push_back(fancyInt.intVal);
		}
		dataset.setTrainImages(readInTrainImages);
		
		vector<unsigned int> readInTestImages;
		for(int i=0;i<dataset.getTestSize();i++){
			file.read(fancyInt.chars,4);
			readInTestImages.push_back(fancyInt.intVal);
		}
		dataset.setTestImages(readInTestImages);
	}
   file.close();
}
//---------------------end:import/export-----------------------------
