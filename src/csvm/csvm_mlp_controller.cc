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

void MLPController::setSettings(MLPSettings s){
	std::cout << "set settings..." << std::endl;
	mlps.reserve(s.nSplitsForPooling * s.nSplitsForPooling);
	
	validationSize = dataset.getTrainSize() * settings.mlpSettings.crossValidationSize;
	trainSize = dataset.getTrainSize() - validationSize;
	amountOfPatchesImage = (settings.datasetSettings.imWidth - settings.scannerSettings.patchWidth + 1) * (settings.datasetSettings.imHeight - settings.scannerSettings.patchHeight + 1);

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

int MLPController::calculateSquareOfPatch(Patch patch){
	int splits = settings.mlpSettings.nSplitsForPooling;
	
	int imWidth  = settings.datasetSettings.imWidth;
	int imHeight = settings.datasetSettings.imHeight;
	
	int middlePatchX = patch.getX() + patch.getWidth()/2;
	int middlePatchY = patch.getY() + patch.getHeight()/2;
	
	int square = middlePatchX / (imWidth/splits) + (splits * (middlePatchY / (imHeight/splits)));
	return square;
}

vector<Feature>& MLPController::createValidationSet(vector<Feature>& validationSet){
	int amountOfImagesCrossVal = dataset.getTrainSize() * settings.mlpSettings.crossValidationSize;

	vector<Patch> patches;   
    
	std::cout << "create validation feature vector... "<< std::endl;
	for(int i = dataset.getTrainSize() - amountOfImagesCrossVal; i < dataset.getTrainSize();i++){
		Image* im = dataset.getTrainImagePtr(i);
		 
		//extract patches
		patches = imageScanner.scanImage(im);
		//std::cout << patches.size() << std::endl;
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

	std::cout << "create random feature vector of size " << nPatches << std::endl;
	
	for(size_t pIdx = 0; pIdx < nPatches; ++pIdx){	  
      Patch patch = imageScanner.getRandomPatch(dataset.getTrainImagePtr(rand() % sizeTrainingSet));
      Feature newFeat = featExtr.extract(patch);
      newFeat.setSquareId(calculateSquareOfPatch(patch));
      
      setMinAndMaxValueNorm(newFeat);
      trainingData.push_back(newFeat);   
   }
	return normalized(trainingData);	
}

void MLPController::createDataBySquares(){
	//trainingSet and validationSet are created so call by reference is possible.
	//since both are large vector's call by reference is faster than call by value
	
	vector<Feature> trainingSet;
 	vector<Feature> validationSet;
	
	minValue = 1000;
	maxValue = 0;

	if(settings.mlpSettings.readInData){
		importFeatureSet(settings.mlpSettings.randomFeatName,trainingSet);
		importFeatureSet(settings.mlpSettings.validationName,validationSet);
	} 
	else {
		trainingSet = createRandomFeatureVector(trainingSet);
		validationSet = createValidationSet(validationSet);
		
		exportFeatureSet("RandomFeat_CIFAR10_50.000_24x24",trainingSet);
		exportFeatureSet("Validation_CIFAR10_50.000_24x24",validationSet);
	}
	
	/*vector<unsigned int> temp = dataset.getTrainImageNums();
	std::cout << dataset.getImagePtr(0) << std::endl;
	exit(-1);
	*/
	
	splitTrain = splitUpDataBySquare(trainingSet);
	splitVal   = splitUpDataBySquare(validationSet);
	
	for(unsigned int i=0;i<mlps.size();i++){
		numPatchesPerSquare.push_back(splitVal[i].size()/validationSize);
	}

	trainingSet.clear();
	validationSet.clear();
}

vector<vector<Feature> > MLPController::splitUpDataBySquare(vector<Feature>& featureSet){
	vector<vector<Feature> > splitBySquares = vector<vector<Feature> >(settings.mlpSettings.nSplitsForPooling * settings.mlpSettings.nSplitsForPooling);
			
	for(unsigned int i = 0;i < featureSet.size();i++){
		splitBySquares[featureSet[i].getSquareId()].push_back(featureSet[i]);	
	}
	return splitBySquares;
}

void MLPController::trainMutipleMLPs(){
	initMLPs();
	
	for(unsigned int i=0;i<mlps.size();i++){ 		
		mlps[i].train(splitTrain[i],splitVal[i],numPatchesPerSquare[i]);
		std::cout << "trained mlp["<<i<<"]\n\n";
    }
    splitTrain.clear();
    splitVal.clear();
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

//---------------------start:import/export---------------------------
union charDouble{
   char chars[8];
   double doubleVal;
};

union charInt{
   char chars[4];
   unsigned int intVal;
};

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
