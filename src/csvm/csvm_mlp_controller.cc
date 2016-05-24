#include <csvm/csvm_mlp_controller.h>
 
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
//--------------start: init MLP's-------------------
void MLPController::setSettings(MLPSettings s){
	cout << "settings set" << std::endl;

	nHiddenBottomLevel = s.nHiddenUnits;
	
	//init global variables	
	nMLPs = pow(s.nSplitsForPooling,2);
	validationSize = dataset.getTrainSize()*s.crossValidationSize;
	trainSize = dataset.getTrainSize() - validationSize;

	for(int i=0;i<2;i++){
		minValues.push_back(1000);
		maxValues.push_back(0);
	}
	
	//reserve global vectors
	numPatchesPerSquare.reserve(nMLPs);
	mlps.reserve(s.stackSize);
	
	mlps = vector<vector<MLPerceptron> >(nMLPs);
	
	for(int j = 0; j < nMLPs;j++){
		MLPerceptron mlp;
		mlp.setSettings(s);
		mlps[0].push_back(mlp);
	}
	
	//settings second parameter	
	MLPerceptron mlp;
	s.nInputUnits = nHiddenBottomLevel * nMLPs;
	s.nHiddenUnits = s.nHiddenUnitsFirstLayer;//find parameter
	mlp.setSettings(s);
	mlps[1].push_back(mlp);
	/*
	for(int i = 0; i < 4;i++){
		std::cout << "memory location of mlp["<<i<<"]: " << &mlps[0][i] << std::endl;
	}
	std::cout << "memory location of first level mlp: " << &mlps[1][0] << std::endl;
	exit(-1);*/
}

void MLPController::setMinAndMaxValueNorm(vector<Feature>& inputFeatures, int index){
	minValues[index] = inputFeatures[0].content[0];
	maxValues[index] = inputFeatures[0].content[0];

	//compute min and max of all the inputs	
	for(unsigned int i = 0; i < inputFeatures.size();i++){
		double possibleMaxValue = *std::max_element(inputFeatures[i].content.begin(), inputFeatures[i].content.end());
		double possibleMinValue = *std::min_element(inputFeatures[i].content.begin(), inputFeatures[i].content.end()); 
	
		if(possibleMaxValue > maxValues[index])
			maxValues[index] = possibleMaxValue;
			
		if(possibleMinValue < minValues[index])
			minValues[index] = possibleMinValue;
	}
}

vector<Feature>& MLPController::normalizeInput(vector<Feature>& inputFeatures, int index){
	if (maxValues[index] - minValues[index] != 0){
		//normalize all the inputs
		for(unsigned int i = 0; i < inputFeatures.size();i++){
			for(int j = 0; j < inputFeatures[i].size;j++)
				inputFeatures[i].content[j] = (inputFeatures[i].content[j] - minValues[index])/(maxValues[index] - minValues[index]);
		}
	}else{
		for(unsigned int i = 0; i<inputFeatures.size();i++){
			for(int j = 0; j < inputFeatures[i].size;j++)
				inputFeatures[i].content[j] = 0;
		}
	}
	return inputFeatures;		
}

int MLPController::calculateSquareOfPatch(Patch patch){
	int splits = settings.mlpSettings.nSplitsForPooling;
	
	int middlePatchX = patch.getX() + patch.getWidth() / 2;
	int middlePatchY = patch.getY() + patch.getHeight() / 2;
	
	int imWidth  = settings.datasetSettings.imWidth;
	int imHeight = settings.datasetSettings.imHeight;
	
	return middlePatchX / (imWidth/splits) + (splits * (middlePatchY / (imHeight/splits)));
}
//-------------------end: init MLP's--------------------------
//-------------------start: data creation methods-------------
/* 
 * First parameter is the feature vector that needs to be filled with complete pictures. All the features of one pictures will behind each other.
 * Second parameter is the start in the training set
 * Third parameter is the end in the trainigset
 * */
vector<Feature>& MLPController::createCompletePictureSet(vector<Feature>& validationSet,int start,int end){
	vector<Patch> patches;   
	for(int i = start; i < end;i++){	
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

vector<vector<Feature> > MLPController::splitUpDataBySquare(vector<Feature>& trainingSet){
	vector<vector<Feature> > splitBySquares = vector<vector<Feature> >(nMLPs);
			
	for(unsigned int i = 0;i < trainingSet.size();i++){
		splitBySquares[trainingSet[i].getSquareId()].push_back(trainingSet[i]);	
	}
	return splitBySquares;
}

//-------------------end: data creation methods---------------
//-------------------start: bottomlevel-----------------------
void MLPController::createDataBottomLevel(vector<vector<Feature> >& splitTrain, vector<vector<Feature> >& splitVal){
	vector<Feature> trainingSet;
 	vector<Feature> validationSet;
	
	splitTrain = splitUpDataBySquare(createRandomFeatureVector(trainingSet));
	splitVal   = splitUpDataBySquare(createCompletePictureSet(validationSet,trainSize,trainSize+validationSize));
	
	for(int i=0;i<nMLPs;i++){
		setMinAndMaxValueNorm(splitTrain[i],0);
	}
	for(int i=0;i<nMLPs;i++){
		numPatchesPerSquare.push_back(splitVal[i].size()/validationSize);
		std::cout<< "numPatchersPerSquare[" << i << "]: " << numPatchesPerSquare[i] << std::endl;
	}
	
	trainingSet.clear();
	validationSet.clear();
}

vector<Feature>& MLPController::createRandomFeatureVector(vector<Feature>& trainingData){
    unsigned int nPatches = settings.scannerSettings.nRandomPatches;
   
	vector<Feature> testData;
	std::cout << "create random feature vector of size: " << nPatches << std::endl;
	
	for(size_t pIdx = 0; pIdx < nPatches; ++pIdx){
		Patch patch = imageScanner.getRandomPatch(dataset.getTrainImagePtr(rand() % trainSize));
		Feature newFeat = featExtr.extract(patch);
		newFeat.setSquareId(calculateSquareOfPatch(patch));
		trainingData.push_back(newFeat);    
	}
	exportFeatureSet("test_trainingdata_featureset",trainingData);
	importFeatureSet("test_trainingdata_featureset");
	exit(-1);
	return trainingData;	
}
//--------------------end: bottom level---------------------------
//--------------------start: first level--------------------------

double Rand(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void MLPController::setFirstLevelData(vector<vector<Feature> >& splitDataBottom,vector<Feature>& dataFirstLevel, int sizeData){
	for(int i = 0; i < sizeData;i++){			
		vector<double> input;
		input.reserve(nHiddenBottomLevel*nMLPs);
		
		for(int j=0;j<nMLPs;j++){			
			vector<Feature>::const_iterator first = splitDataBottom[j].begin()+(numPatchesPerSquare[j]*i);
			vector<Feature>::const_iterator last = splitDataBottom[j].begin()+(numPatchesPerSquare[j]*(i+1));

			vector<double> inputTemp = vector<double>(nHiddenBottomLevel,-10.0);
			
			mlps[0][j].returnHiddenActivation(vector<Feature>(first,last),inputTemp);
			input.insert(input.end(),inputTemp.begin(),inputTemp.end());
		}
		/*
		for(unsigned int j =0;j<input.size();j++)
			if(input[j] == 0)
				input[j] = 0.1;
		*/
		Feature newFeat = new Feature(input);	
		newFeat.setLabelId(splitDataBottom[0][i*numPatchesPerSquare[0]].getLabelId());
		
		dataFirstLevel.push_back(newFeat);
	}
}
/* 
 * In the first level the mlp training is now based on complete images. 
 * This is different from the bottom level where it is based on random patches 
 * */
void MLPController::createDataFirstLevel(vector<Feature>& inputTrainFirstLevel, vector<Feature>& inputValFirstLevel){
  	vector<Feature> trainingSet;
	vector<Feature> validationSet;
	
	//increasing the stride to decrease the size of the complete picture set
	imageScanner.setScannerStride(settings.mlpSettings.scanStrideFirstLayer);
	
	vector<vector<Feature> > splitTrain = splitUpDataBySquare(createCompletePictureSet(trainingSet,0,trainSize));
	vector<vector<Feature> > splitVal   = splitUpDataBySquare(createCompletePictureSet(validationSet,trainSize,trainSize+validationSize));
	
	//set new number of patches per square
	for(int i=0;i<nMLPs;i++){
		numPatchesPerSquare[i] = splitVal[i].size()/validationSize;
		std::cout<< "numPatchersPerSquare with stride increase[" << i << "]: " << numPatchesPerSquare[i] << std::endl;
	}
	
	trainingSet.clear();
	validationSet.clear();
	
	for(int i=0;i<nMLPs;i++){
		normalizeInput(splitTrain[i],0);
		normalizeInput(splitVal[i],0);
	}
	
	setFirstLevelData(splitTrain,inputTrainFirstLevel,trainSize); 	//set training set
	setFirstLevelData(splitVal,inputValFirstLevel,validationSize); 	//set validation set
    
	setMinAndMaxValueNorm(inputTrainFirstLevel,1);    //set min and max value for first level normalization
}
//-----------start: training MLP's-------------------------
void MLPController::trainMutipleMLPs(){
	vector<vector<Feature> > splitTrain;
	vector<vector<Feature> > splitVal;
	
	//TODO: 
	/*
	 * try to export splitTrain and splitVal for different kinds of patchs size and randomfeature sizes.
	 * This could really help in speeding up the training phase from the mlp.
	 * */
	createDataBottomLevel(splitTrain,splitVal);
	
	for(int i=0;i<nMLPs;i++){
		normalizeInput(splitTrain[i],0);
		normalizeInput(splitVal[i],0);
	}
	
	for(int i=0;i<nMLPs;i++){ 		
		mlps[0][i].train(splitTrain[i],splitVal[i],numPatchesPerSquare[i]);
		//std::cout << "mlp["<<i<<"] from level 0 finished training" << std::endl << std::endl;
    }
    
    std::cout << "create training data for first level... " << std::endl;
    
    vector<Feature> inputTrainFirstLevel;
	vector<Feature> inputValFirstLevel;
    
    createDataFirstLevel(inputTrainFirstLevel,inputValFirstLevel);
    
    //std::cout << "min value first level: " << minValues[1] << std::endl;
    //std::cout << "max value first level: " << maxValues[1] << std::endl;
    
    normalizeInput(inputTrainFirstLevel,1);
    normalizeInput(inputValFirstLevel,1);
    
    /*for(unsigned int i = 0; i < inputTrainFirstLevel.size();i++){
		std::cout << "feature ["<<i<<"]: " << inputTrainFirstLevel[i].getLabelId() << std::endl;
		for(int j = 0; j < inputTrainFirstLevel[i].size;j++){
			std::cout << inputTrainFirstLevel[i].content[j] << ", ";
		}
		std::cout << std::endl << std::endl;
	}
	*/
	//numPatches is now 1, because training is not done with randompatches anymore.
	//Training is done one complete images and so is the validation.
	mlps[1][0].train(inputTrainFirstLevel,inputValFirstLevel,1); 
	std::cout << "mlp[0] from level 1 finished training" << std::endl;
}
//--------------end: training MLP's-----------------------------
//-------------start: MLP classification------------------------
union charDouble{
   char chars[8];
   double doubleVal;
};

union charInt{
   char chars[4];
   unsigned int intVal;
};

unsigned int MLPController::mlpMultipleClassify(Image* im){
	int numOfImages = 1;
	
	vector<Patch> patches;
	vector<Feature> dataFeatures;

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
	
	vector<vector<Feature> > testFeaturesBySquare = splitUpDataBySquare(normalizeInput(dataFeatures,0)); //split test features by square	
	vector<Feature> testDataFirstLevel; //empty feature vector that will be filled with first level features
	setFirstLevelData(testFeaturesBySquare,testDataFirstLevel,numOfImages);
	/*
	for(int i=0;i<testDataFirstLevel[0].size;i++){
		std::cout << testDataFirstLevel[0].content[i] << ", "; 
	}
	std::cout << std::endl;
	*/
	normalizeInput(testDataFirstLevel,1); 
		
	//for(int i=0;i<10;i++){
	//	std::cout << votingHisto[i] << std::endl;
	//}
	//std::cout << std::endl;
	
	//std::cout << "answer: " << answer << std::endl;
	
	int answer = mlps[1][0].classify(testDataFirstLevel);
	return answer;
}
//---------------------end:MLP classification------------------------
//---------------------start:import/export---------------------------
void MLPController::exportFeatureSet(string filename, vector<Feature>& featureSet){
   /* Featureset file conventions:
    * 
    * first,  Dataset			: 0, CIFAR10 1,MNIST   (1 byte)
    * second, Amount of features: 0-10.000.000         (4 bytes)
    * third,  PatchWidth		: 0-36                 (1 byte)
    * fourth, PatchHeigth		: 0-36                 (1 byte)  
    * fifth,  FeatSize			: 0-10.000             (1 byte)
    * sixth,  FeatExtractor		: 0,HOG 1,CLEAN        (1 byte)  
    * 
    * Now each lines consists all the double values in the feature
    *  
    *  No seperator characters are used
   */
   
   charInt fancyInt;
   charDouble fancyDouble;
   
   //cout << "\t\twordSize:\t" << bow[0].content.size() << "\n\tfilename:\t" << filename.c_str() << endl;
   //unsigned int wordSize = bow[0].content.size();
   ofstream file(filename.c_str(),  ios::binary);
   /*
   //write dataset used
	stringstream ss;
	ss << settings.datasetSettings.type;
	fancyInt.intVal = (ss.str() == "CIFAR10" ? 0 : 1); //is this correct?
	file.write(fancyInt.chars, 4);
	ss.clear();
   */
   //write amount of features
   fancyInt.intVal = settings.scannerSettings.nRandomPatches;
   file.write(fancyInt.chars, 1);
   /*
   //write patchWidth
   fancyInt.intVal = settings.scannerSettings.patchWidth;
   file.write(fancyInt.chars, 1);
   
   //write patchWidth
   fancyInt.intVal = settings.scannerSettings.patchHeight;
   file.write(fancyInt.chars, 1);
   
   //write FeatSize
   fancyInt.intVal = settings.mlpSettings.nInputUnits;
   file.write(fancyInt.chars, 1);   
   
   //write FeatExtractor method
   ss << settings.featureSettings.featureType;
   fancyInt.intVal = (ss.str() == "HOG" ? 0 : 1);
   file.write(fancyInt.chars, 1); 
   */
   /*
	for(unsigned int i=0;i<settings.scannerSettings.nRandomPatches;i++){
		for(int j=0;j<settings.mlpSettings.nInputUnits;j++){
			fancyDouble.doubleVal = featureSet[i].content[j];
			file.write(fancyDouble.chars, 8);
		}
	}
   */
   file.close();
}

void MLPController::importFeatureSet(string filename){
   charInt fancyInt;
   charDouble fancyDouble;
   unsigned int typesize;
   unsigned int featDims;
   
   ifstream file(filename.c_str(), ios::binary);
   
   file.read(fancyInt.chars,1);
   std::cout << fancyInt.intVal << std::endl;
   
   //std::cout << file.read(fancyInt.chars,4) << std::endl;
   //std::cout << file.read(fancyInt.chars,1) << std::endl;
   //std::cout << file.read(fancyInt.chars,1) << std::endl;
   
   file.close();
}
//---------------------end:import/export-----------------------------
