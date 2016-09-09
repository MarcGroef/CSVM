#include <csvm/csvm_mlp_controller.h>
 
/* This class will control the multiple MLPs for mlp pooling
 */
 
 using namespace std;
 using namespace csvm;
 
MLPController::MLPController(FeatureExtractor* fe, ImageScanner* imScan, CSVMSettings* se, CSVMDataset* ds){
	featExtr = *fe;
	imageScanner = *imScan;
	settings = *se;
	dataset = ds;
}
//--------------start: init MLP's-------------------
void MLPController::setSettings(MLPSettings s){
	cout << "settings set" << std::endl;

	nHiddenBottomLevel = s.nHiddenUnits;
	first = 1;
	//init global variables	
	nMLPs = pow(s.nSplitsForPooling,2);
	validationSize = dataset->getTrainSize()*s.crossValidationSize;
	
	trainSize = dataset->getTrainSize() - validationSize;
	
	trainSizeBottomLevel = trainSize/2; //0-trainSize/2
	trainSizeFirstLevel = trainSize/2;  //trainSize/2 - trainSize
	
	amountOfPatchesImage = (settings.datasetSettings.imWidth - settings.scannerSettings.patchWidth + 1) * (settings.datasetSettings.imHeight - settings.scannerSettings.patchHeight + 1);

	for(int i=0;i<2;i++){
		minValues.push_back(1000);
		maxValues.push_back(0);
	}
	
	//reserve global vectors
	numPatchesPerSquare.reserve(nMLPs);
	mlps.reserve(s.stackSize);
	
	mlps = vector<vector<MLPerceptron> >(s.stackSize);
	
	for(int j = 0; j < nMLPs;j++){
		MLPerceptron mlp;
		mlp.setSettings(s);
		mlps[0].push_back(mlp);
	}
	//settings second parameter	
	if(s.stackSize == 2){
		MLPerceptron mlp;
		s.nInputUnits = nHiddenBottomLevel * nMLPs;
		s.nHiddenUnits = s.nHiddenUnitsFirstLayer;//find parameter
		mlp.setSettings(s);
		mlps[1].push_back(mlp);
	}
	/*
	for(int i = 0; i < 4;i++){
		std::cout << "memory location of mlp["<<i<<"]: " << &mlps[0][i] << std::endl;
	}
	std::cout << "memory location of first level mlp: " << &mlps[1][0] << std::endl;
	exit(-1);
	*/
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

void MLPController::changeRange(vector<Feature>& data, double newMin, double newMax){
	double max = 0.0;
	double min = 2.0;
	
	for(unsigned int i=0; i<data.size(); i++){
	  for(int j=0; j< data[i].size;j++){
		if(data[i].content[j] > max)
			max = data[i].content[j];
		if(data[i].content[j] < min)
			min = data[i].content[j];
	  }
	}
	
	for(unsigned int i=0;i<data.size();i++){
	  for(int j=0; j<data[i].size; j++){
		  data[i].content[j] = (data[i].content[j] - min) * ((newMax-newMin)/(max-min)) + newMin;
	  }
	}
}
//-------------------end: init MLP's--------------------------
//-------------------start: data creation methods-------------
/* 
 * First parameter is the feature vector that needs to be filled with complete pictures. All the features of one pictures will behind each other.
 * Second parameter is the start in the training set
 * Third parameter is the end in the trainigset
 * */
vector<Feature>& MLPController::createCompletePictureSet(vector<Feature>& validationSet, int start, int end){
	vector<Patch> patches;   
	for(int i = start; i < end;i++){	
		Image* im = dataset->getTrainImagePtr(i);
		 
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

	if(settings.mlpSettings.readInData){
		importFeatureSet(settings.mlpSettings.readRandomFeatName,trainingSet);
		importFeatureSet(settings.mlpSettings.readValidationName,validationSet);
	} 
	else {
		trainingSet = createRandomFeatureVector(trainingSet);
		validationSet = createCompletePictureSet(validationSet,trainSize,trainSize+validationSize);
		
		/*
		for(int i=0;i<trainingSet.size();i++){
		  for(int j=0;j<trainingSet[i].size;j++){
		  std::cout << trainingSet[i].content[j] << ", ";
		  } 
		    cout << endl << endl;
		 }
		  exit(-1);
		  */
	 
		
		setMinAndMaxValueNorm(trainingSet,0);

		normalizeInput(trainingSet,0);
		normalizeInput(validationSet,0);
	}
	
	if(settings.mlpSettings.saveData){
		exportFeatureSet(settings.mlpSettings.saveRandomFeatName,trainingSet);
		exportFeatureSet(settings.mlpSettings.saveValidationName,validationSet);
	}
	
	//Change the range of the data
	//changeRange(trainingSet,0,0.5);
	//changeRange(validationSet,-0.25,0.25);
  /*
	for(int i=0;i<trainingSet.size();i++){
	  for(int j=0;j<trainingSet[i].size;j++){
	    std::cout << trainingSet[i].content[j] << ", ";
	  } 
	  cout << endl << endl;
	}
	 exit(-1);
  */	 
	splitTrain = splitUpDataBySquare(trainingSet);
	splitVal   = splitUpDataBySquare(validationSet);
	
	for(int i=0;i<nMLPs;i++){
		numPatchesPerSquare.push_back(splitVal[i].size()/validationSize);
		std::cout<< "numPatchersPerSquare["<<i<<"]: " << numPatchesPerSquare[i] << std::endl;
	} cout << endl;
	/*
	for(int i=0;i<nMLPs;i++){
	  cout << "size of random feature vector[" <<i<<"]: " << splitTrain[i].size() << endl;
	}*/
	
	trainingSet.clear();
	validationSet.clear();
}

vector<Feature>& MLPController::createRandomFeatureVector(vector<Feature>& trainingData){
    unsigned int nPatches = settings.scannerSettings.nRandomPatches;
   
	vector<Feature> testData;
	std::cout << "create random feature vector of size: " << nPatches << std::endl;
	
	for(size_t pIdx = 0; pIdx < nPatches; ++pIdx){
		Patch patch = imageScanner.getRandomPatch(dataset->getTrainImagePtr(rand() % trainSize));
		Feature newFeat = featExtr.extract(patch);
		newFeat.setSquareId(calculateSquareOfPatch(patch));
		trainingData.push_back(newFeat);    
	}
	
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
		vector<double> inputVector;
		inputVector.reserve(nHiddenBottomLevel*nMLPs);
		
		for(int j=0;j<nMLPs;j++){			
			vector<Feature>::const_iterator first = splitDataBottom[j].begin()+(numPatchesPerSquare[j]*i);
			vector<Feature>::const_iterator last = splitDataBottom[j].begin()+(numPatchesPerSquare[j]*(i+1));
			
			vector<double> hiddenActivationSquare = mlps[0][j].returnHiddenActivationToMethod(vector<Feature>(first,last));
			
			inputVector.insert(inputVector.end(),hiddenActivationSquare.begin(),hiddenActivationSquare.end());
			
			//Is this saver? than above?
			//for(unsigned int i=0;i<hiddenActivationSquare.size();i++)
			//	inputVector.push_back(hiddenActivationSquare[i]);
		}
		
		Feature newFeat = new Feature(inputVector);	
		//get label abritrary patch
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
	
	trainingSet = createCompletePictureSet(trainingSet,0,trainSize);
	validationSet = createCompletePictureSet(validationSet,trainSize,trainSize+validationSize);
	
	normalizeInput(trainingSet,0);
	normalizeInput(validationSet,0);
	
	vector<vector<Feature> > splitTrain = splitUpDataBySquare(trainingSet);
	vector<vector<Feature> > splitVal   = splitUpDataBySquare(validationSet);
	
	//set new number of patches per square
	for(int i=0;i<nMLPs;i++){
		numPatchesPerSquare[i] = splitVal[i].size()/validationSize;
		std::cout<< "numPatchersPerSquare with stride increase[" << i << "]: " << numPatchesPerSquare[i] << std::endl;
	} cout << endl;
	
	trainingSet.clear();
	validationSet.clear();
	
	setFirstLevelData(splitTrain,inputTrainFirstLevel,trainSize); 	//set training set
	setFirstLevelData(splitVal,inputValFirstLevel,validationSize); 	//set validation set
    
	cout << "size of input vector first level: " << inputTrainFirstLevel.size() << endl;
	
	setMinAndMaxValueNorm(inputTrainFirstLevel,1); //set min and max value for first level normalization
	
	normalizeInput(inputTrainFirstLevel,1);
	normalizeInput(inputValFirstLevel,1);
	
	//Change the range of the data from 0,1 to -1,1
	//changeRange(inputTrainFirstLevel,0,0.5);
	//changeRange(inputValFirstLevel,0,0.5);
}
//-----------start: training MLP's-------------------------
void MLPController::trainMutipleMLPs(){
	vector<vector<Feature> > splitTrain;
	vector<vector<Feature> > splitVal;
	
	if(!settings.mlpSettings.readMLP){
	createDataBottomLevel(splitTrain,splitVal);
		
	for(int i=0;i<nMLPs;i++){ 		
		mlps[0][i].train(splitTrain[i],splitVal[i],numPatchesPerSquare[i]);
		std::cout << "mlp["<<i<<"] from level 0 finished training on randomfeat" << std::endl << std::endl;
	}
		/*
		for(int i=0;i<nMLPs;i++){ 		
			mlps[0][i].train(splitVal[i],numPatchesPerSquare[i]);
			std::cout << "mlp["<<i<<"] from level 0 finished training on validation set" << std::endl << std::endl;
		}
		*/
	} else 
	    importPreTrainedMLP(settings.mlpSettings.readMLPName);
	if(settings.mlpSettings.saveMLP)
	  exportTrainedMLP(settings.mlpSettings.saveMLPName);
    //std::cout << "create training data for first level... " << std::endl;
    if(settings.mlpSettings.stackSize == 2){
		vector<Feature> inputTrainFirstLevel;
		vector<Feature> inputValFirstLevel;
    
		createDataFirstLevel(inputTrainFirstLevel,inputValFirstLevel);
    
		//std::cout << "min value first level: " << minValues[1] << std::endl;
		//std::cout << "max value first level: " << maxValues[1] << std::endl;
    
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
}
//--------------end: training MLP's-----------------------------
//-------------start: MLP classification------------------------

void MLPController::dropOutTesting(vector<vector<vector<double> > >& newWeights){
  double p = 0.5;
  for(unsigned int i=0;i<newWeights.size();i++){
    for(unsigned int  j=0;j<newWeights[i].size();j++){
      for(unsigned int k=0;k<newWeights[i][j].size();k++){
	 newWeights[i][j][k] *= p;
      }
    }
  }
} 
unsigned int MLPController::mlpMultipleClassify(Image* im){
	int numOfImages = 1;
	int answer = -1;
	
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
	dataFeatures = normalizeInput(dataFeatures,0);
	//changeRange(dataFeatures,-0.25,0.25);

	/*for(int i=0;i<dataFeatures.size();i++){
		for(int j=0;j<dataFeatures[i].size;i++){
			cout << dataFeatures[i].content[j] << ", ";
			} cout << endl;
		}
		exit(-1);
		*/
	vector<vector<Feature> > testFeaturesBySquare = splitUpDataBySquare(dataFeatures); //split test features by square	
	
	if(settings.mlpSettings.stackSize == 1){
	/*  if(first == 1){
	    vector<vector<vector<double > > > newWeights;
	    
	    for(int i=0;i<nMLPs;i++){
	     newWeights = mlps[0][i].getWeightMatrix();
	     dropOutTesting(newWeights);
	     mlps[0][i].setWeightMatrix(newWeights);
	    }
	    first=0;
	  }*/
		vector<double> votingHistogram = vector<double>(settings.mlpSettings.nOutputUnits,0.0);
		vector<double> outputProp;
		for(int i=0;i<nMLPs;i++){
			outputProp = mlps[0][i].classifyPooling(testFeaturesBySquare[i]);
			for(int j=0;j<settings.mlpSettings.nOutputUnits;j++){
				votingHistogram[j] += outputProp[j];
			}
		}
		double highestProp = 0;
		int mostVotedClass = -1;
		for(int i=0;i<settings.mlpSettings.nOutputUnits;i++){
			if(votingHistogram[i] > highestProp){
				highestProp = votingHistogram[i];
				mostVotedClass = i;
			}
		}
		answer = mostVotedClass;
	}
	
	if(settings.mlpSettings.stackSize == 2){
		vector<Feature> testDataFirstLevel; //empty feature vector that will be filled with first level features
		setFirstLevelData(testFeaturesBySquare,testDataFirstLevel,numOfImages);
		/*
		for(int i=0;i<testDataFirstLevel[0].size;i++){
			std::cout << testDataFirstLevel[0].content[i] << ", "; 
		}
		std::cout << std::endl;
		*/
	  /*if(first == 1){
	    vector<vector<vector<double > > > newWeights;
	    
	    for(int i=0;i<nMLPs;i++){
	     newWeights = mlps[0][i].getWeightMatrix();
	     dropOutTesting(newWeights);
	     mlps[0][i].setWeightMatrix(newWeights);
	    }
	    
	    newWeights = mlps[1][0].getWeightMatrix();
	    dropOutTesting(newWeights);
	    mlps[1][0].setWeightMatrix(newWeights);
	    
	    first=0;
	  }*/
		normalizeInput(testDataFirstLevel,1); 
		//changeRange(dataFeatures,0.0,0.5);
		/*
		for(int i=0;i<testDataFirstLevel.size();i++){
			for(int j=0;j<testDataFirstLevel[i].size;j++){
				cout << testDataFirstLevel[i].content[j] << ", ";
				} cout << endl;
			}
			exit(-1);
		*/
		
		//for(int i=0;i<10;i++){
		//	std::cout << votingHisto[i] << std::endl;
		//}
		//std::cout << std::endl;
		
		//std::cout << "answer: " << answer << std::endl;
		
		answer = mlps[1][0].classify(testDataFirstLevel);
	}
	return answer;
}
//---------------------end:MLP classification------------------------
//---------------------start:import/export---------------------------
union charDouble{
   char chars[8];
   double doubleVal;
};

union charInt{
   char chars[4];
   unsigned int intVal;
};

void MLPController::importPreTrainedMLP(string filename){
	charInt fancyInt;
	charDouble fancyDouble;
	
	ifstream file(filename.c_str(), ios::binary);
	
	file.read(fancyInt.chars,4);
	int readInInputUnits = fancyInt.intVal;
   	if(readInInputUnits != settings.mlpSettings.nInputUnits){
		std::cout << "The input units that is read in is " << readInInputUnits << " and in the settings file you have " << settings.mlpSettings.nInputUnits << ", please change this" << std::endl;
		exit(-1);
	}
	
	file.read(fancyInt.chars,4);
	int readInHiddenUnits = fancyInt.intVal;
   	if(readInHiddenUnits != settings.mlpSettings.nHiddenUnits){
		std::cout << "The hidden units that is read in is " << readInHiddenUnits << " and in the settings file you have " << settings.mlpSettings.nHiddenUnits << ", please change this" << std::endl;
		exit(-1);
	}
	
	file.read(fancyInt.chars,4);
	int readInOutputUnits = fancyInt.intVal;
   	if(readInOutputUnits != settings.mlpSettings.nOutputUnits){
		std::cout << "The output units that is read in is " << readInOutputUnits << " and in the settings file you have " << settings.mlpSettings.nOutputUnits << ", please change this" << std::endl;
		exit(-1);
	}
	
	file.read(fancyInt.chars,4);
	int readInNSplitsForPooling = fancyInt.intVal;
   	if(readInNSplitsForPooling != settings.mlpSettings.nSplitsForPooling){
		std::cout << "The nSplitsForPooling that is read in is " << readInNSplitsForPooling << " and in the settings file you have " << settings.mlpSettings.nSplitsForPooling << ", please change this" << std::endl;
		exit(-1);
	}
	
	file.read(fancyInt.chars,4);
	unsigned int datasetNum = fancyInt.intVal; //some check that his num is smaller than 2
	CSVMDatasetType readInDatasetType = DATASET_CIFAR10; //set to CIFAR10 to make the compiler stop complaining
   
   	switch(datasetNum){
      case 0:
         readInDatasetType = DATASET_CIFAR10;
         break;
      case 1:
	readInDatasetType = DATASET_MNIST;
	break;
      }
   
	if(readInDatasetType != dataset->getType()){
		std::cout << "The dataset that is read in is " << readInDatasetType << " and in the settings file you have " << dataset->getType() << ", please change this" << std::endl;
		exit(-1);
	}
	//read min value first level
	file.read(fancyDouble.chars,8);
	minValues[0] = fancyDouble.doubleVal;
	
	//read max value first level
	file.read(fancyDouble.chars,8);
	maxValues[0] = fancyDouble.doubleVal;
	
	int maxUnits=0;
	
	if(maxUnits < settings.mlpSettings.nInputUnits){
	  maxUnits = settings.mlpSettings.nInputUnits;
	}
	if(maxUnits < settings.mlpSettings.nHiddenUnits){
	  maxUnits = settings.mlpSettings.nHiddenUnits;
	}
	if(maxUnits < settings.mlpSettings.nOutputUnits){
	  maxUnits = settings.mlpSettings.nOutputUnits;
	}
	
	vector<vector<double> > biasNodes = vector<vector<double> >(settings.mlpSettings.nLayers-1,vector<double>(maxUnits,0.0));
	vector<vector<vector<double> > > weights = vector<vector<vector<double> > >(settings.mlpSettings.nLayers-1,vector<vector<double> >(maxUnits,vector<double>(maxUnits,0.0)));
	
	for(int i=0; i<nMLPs;i++){ //amount of mlps that needs to be read in
		for(int j=0;j<settings.mlpSettings.nLayers-1;j++){ //amount of weight vectors
			for(int k=0;k<maxUnits;k++){ //amount of collums
				for(int l=0;l<maxUnits;l++){ // amount of rows
				  file.read(fancyDouble.chars,8);
				  weights[j][k][l] = fancyDouble.doubleVal;
					
				}
			}
		}
		for(int j=0;j<settings.mlpSettings.nLayers-1;j++){ //amount of bais nodes matrixs
			for(int k=0;k<maxUnits;k++){ // max amount of units, there is only one column and 
				file.read(fancyDouble.chars,8);
				biasNodes[j][k] = fancyDouble.doubleVal;
			}
		}
	//load the weights and bais matrix into the mlp
	mlps[0][i].loadInMLP(weights,biasNodes);
	
	//reset weight matrixes for the next mlp
	biasNodes = vector<vector<double> >(settings.mlpSettings.nLayers-1,vector<double>(maxUnits,0.0));
	weights = vector<vector<vector<double> > >(settings.mlpSettings.nLayers-1,vector<vector<double> >(maxUnits,vector<double>(maxUnits,0.0)));
	
	}
}

void MLPController::exportTrainedMLP(string filename){
	/*
	 * First,  nInputUnits : 0 - maxInt (4 bytes)
	 * Second, nHiddenUnits: 0 - maxInt (4 bytes)
	 * Third,  nOutputUnits: 0 - maxInt usually 10 (4 bytes)
	 * Fourth, nSplitsForPooling : 1-2 (4 bytes)
	 * Fifth,  Dataset     : CIFAR10=0, MNIST=1 (4 bytes)
	 * 
	 * Save weight and bias matrix.
	 * 
	 * No seperator char is used 
	 * 
*/	 
	 charInt fancyInt;
	 charDouble fancyDouble;
	 
	 ofstream file(filename.c_str(), ios::binary);
	 
	 //write nInputUnits
	 fancyInt.intVal = settings.mlpSettings.nInputUnits;
	 file.write(fancyInt.chars, 4);
	 
	 //write nHiddenUnits
	 fancyInt.intVal = settings.mlpSettings.nHiddenUnits;
	 file.write(fancyInt.chars, 4);
	 
	 //write nOutputUnits
	 fancyInt.intVal = settings.mlpSettings.nOutputUnits;
	 file.write(fancyInt.chars, 4);
	 
	 //write nSplitsForPooling
	 fancyInt.intVal = settings.mlpSettings.nSplitsForPooling;
	 file.write(fancyInt.chars, 4);
	 
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
	
	 //write min value for first layer
	 fancyDouble.doubleVal = minValues[0]; 
	file.write(fancyDouble.chars, 8);
	
	//write max value for first layer
	fancyDouble.doubleVal = maxValues[0];
	file.write(fancyDouble.chars, 8);
	 
	for(int i=0; i<nMLPs;i++){ //amount of mlps
		vector<vector<double> > biasNodes = mlps[0][i].getBiasNodes();
		vector<vector<vector<double> > > weights = mlps[0][i].getWeightMatrix();
		
		for(unsigned int j=0;j<weights.size();j++){ //amount of weight vectors, 2 for 3 layer mlp
			for(unsigned int k=0;k<weights[j].size();k++){ //size of the max amount of nodes of the mlp
				for(unsigned int l=0;l<weights[j][k].size();l++){ //size of the max amount of nodes of the mlp
					fancyDouble.doubleVal = weights[j][k][l];
					file.write(fancyDouble.chars, 8);
				}
			}
		}
		for(unsigned int j=0;j<biasNodes.size();j++){ //amount of bias node vectors
			for(unsigned int k=0;k<biasNodes[j].size();k++){ //size of max amount of nodes of the mlp
				fancyDouble.doubleVal = biasNodes[j][k];
				file.write(fancyDouble.chars, 8);
			}
		}
	}
}

void MLPController::exportFeatureSet(string filename, vector<Feature>& featureVector){
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
		//write min value from bottom level
		fancyDouble.doubleVal = minValues[0]; 
		file.write(fancyDouble.chars, 8);
		//std::cout << "min value in write: " << fancyDouble.doubleVal << std::endl;
		//write max value
		fancyDouble.doubleVal = maxValues[0];
		file.write(fancyDouble.chars, 8);
		//std::cout << "min value in write: " << fancyDouble.doubleVal << std::endl;

		vector<unsigned int> trainImages = dataset->getTrainImageNums();
		for(int i=0;i<dataset->getTrainSize();i++){
			fancyInt.intVal = trainImages[i];
			//std::cout << trainImages[i] << std::endl;
			file.write(fancyInt.chars,4);
		}
		vector<unsigned int> testImages = dataset->getTestImageNums();
		for(int i=0;i<dataset->getTestSize();i++){
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
   
	if(readInDatasetType != dataset->getType()){
		std::cout << "The dataset that is read in is " << readInDatasetType << " and in the settings file you have " << dataset->getType() << ", please change this" << std::endl;
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
		sizeOfFeatureVector = dataset->getTrainSize() * settings.mlpSettings.crossValidationSize * amountOfPatchesImage;
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
		minValues[0] = fancyDouble.doubleVal;
	
		//read max value
		file.read(fancyDouble.chars, 8);
		maxValues[0] = fancyDouble.doubleVal;
		
		vector<unsigned int> readInTrainImages;
		for(int i=0;i<dataset->getTrainSize();i++){
			file.read(fancyInt.chars,4);
			//std::cout << "imageNum["<<i<<"]: " << fancyInt.intVal << std::endl;
			readInTrainImages.push_back(fancyInt.intVal);
		}
		dataset->setTrainImages(readInTrainImages);
		
		vector<unsigned int> readInTestImages;
		for(int i=0;i<dataset->getTestSize();i++){
			file.read(fancyInt.chars,4);
			readInTestImages.push_back(fancyInt.intVal);
		}
		dataset->setTestImages(readInTestImages);
	}
   file.close();
}
//---------------------end:import/export-----------------------------
