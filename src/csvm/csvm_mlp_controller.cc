#include <csvm/csvm_mlp_controller.h>
 
/* This class will control the multiple MLPs for mlp pooling
 */
 
 using namespace std;
 using namespace csvm;

MLPController::MLPController(FeatureExtractor* fe, ImageScanner* imScan,CSVMDataset* ds){
	featExtr = *fe;
	imageScanner = *imScan;
	dataset = ds;
}
//--------------start: init MLP's-------------------
void MLPController::setSettings(controllerSettingsPack controller){
    cout << "settings set" << endl;    
    this->controller = controller;
    this->controllerSettings = controller.controllerSettings;
    this->mlpSettings = controller.mlpSettings;
    
    
    
    nMLPs = pow(controllerSettings.nSplitsForPooling,2);
    
    validationSize = dataset->getTrainSize()*mlpSettings.crossValidationSize;
    trainSize = dataset->getTrainSize()-validationSize;
	
    first = 1;
    //60% of the train data is for the bottom level mlp and the other 40% is for the first level.
    trainSizeBottomLevel = trainSize*0.6; 
    trainSizeFirstLevel = trainSize*0.4; 
    
    if(controllerSettings.splitTrainSet){
            cout << "The trainingSet is split for the bottom and the first level" << endl;
            cout << "The bottom level will be trained with patches from " << trainSizeBottomLevel << " images" << endl;
            cout << "The first level will use " << trainSizeFirstLevel << " images" << endl;
    }

    amountOfPatchesImage = (controller.datasetSettings.imWidth - controller.scannerSettings.patchWidth) * (controller.datasetSettings.imHeight - controller.scannerSettings.patchHeight);
        
    for(int i=0;i<2;i++){
		minValues.push_back(1000);
		maxValues.push_back(0);
    }
    
    //reserve global vectors
    numPatchesPerSquare.reserve(nMLPs);
    mlps.reserve(controllerSettings.stackSize);
    
    mlps = vector<vector<MLPerceptron> >(controllerSettings.stackSize);
    
    for(int j = 0; j < nMLPs;j++){
            MLPerceptron mlp;
            mlp.setSettings(mlpSettings);
            mlps[0].push_back(mlp);
    }
    //settings second parameter	
    if(controllerSettings.stackSize == 2){
            MLPerceptron mlp;
            firstLevelMLP = mlpSettings;
            firstLevelMLP.nInputUnits = mlpSettings.nHiddenUnits * nMLPs;
            firstLevelMLP.nHiddenUnits = controllerSettings.nHiddenUnitsFirstLayer;//find parameter
            mlp.setSettings(firstLevelMLP);
            mlps[1].push_back(mlp);
    }
    /*
    for(int i = 0; i < 4;i++){
            cout << "memory location of mlp["<<i<<"]: " << &mlps[0][i] << endl;
    }
    cout << "memory location of first level mlp: " << &mlps[1][0] << endl;
    exit(-1);
    */
    //TODO maybe
    //set poolingType to array
    //controllerSettings.poolingType;
}

void MLPController::setMinAndMaxValueNorm(vector<Feature>& inputFeatures, int index){
	minValues[index] = inputFeatures[0].content[0];
	maxValues[index] = inputFeatures[0].content[0];

	//compute min and max of all the inputs	
	for(unsigned int i = 0; i < inputFeatures.size();i++){
		double possibleMaxValue = *max_element(inputFeatures[i].content.begin(), inputFeatures[i].content.end());
		double possibleMinValue = *min_element(inputFeatures[i].content.begin(), inputFeatures[i].content.end()); 
	
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
	int splits = controllerSettings.nSplitsForPooling;
	
	int middlePatchX = patch.getX() + patch.getWidth() / 2;
	int middlePatchY = patch.getY() + patch.getHeight() / 2;
	
	int imWidth  = controller.datasetSettings.imWidth;
	int imHeight = controller.datasetSettings.imHeight;
	
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

vector<Feature> MLPController::createTestSet(){
        vector<Feature> testSet;
    	vector<Patch> patches; 
        
        int testSetSize = dataset->getTestSize();
	
        for(int i = 0; i < testSetSize;i++){	
		Image* im = dataset->getTestImagePtr(i);
		 
		//extract patches
		patches = imageScanner.scanImage(im);
      
		//extract features from all patches
		for(size_t patch = 0; patch < patches.size(); ++patch){
			Feature newFeat = featExtr.extract(patches[patch]);
			newFeat.setSquareId(calculateSquareOfPatch(patches[patch]));
			testSet.push_back(newFeat);
		}
	}
	return testSet;
}

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
        
	if(controllerSettings.readInData){
		importFeatureSet(controllerSettings.readRandomFeatName,trainingSet);
		importFeatureSet(controllerSettings.readValidationName,validationSet);
	} 
	else {
                trainingSet = createRandomFeatureVector(trainingSet);
                validationSet = createCompletePictureSet(validationSet,trainSize,trainSize+validationSize);
                

		setMinAndMaxValueNorm(trainingSet,0);

		normalizeInput(trainingSet,0);
		normalizeInput(validationSet,0);
	}
	
	if(controllerSettings.saveData){
		exportFeatureSet(controllerSettings.saveRandomFeatName,trainingSet);
		exportFeatureSet(controllerSettings.saveValidationName,validationSet);
	}
	
	//Change the range of the data
	//changeRange(trainingSet,0,1);
	//changeRange(validationSet,0,1);
  /*
	for(int i=0;i<trainingSet.size();i++){
	  for(int j=0;j<trainingSet[i].size;j++){
	    cout << trainingSet[i].content[j] << ", ";
	  } 
	  cout << endl << endl;
	}
	 exit(-1);
  */	 
	splitTrain = splitUpDataBySquare(trainingSet);
	splitVal   = splitUpDataBySquare(validationSet);
	
	for(int i=0;i<nMLPs;i++){
		numPatchesPerSquare.push_back(splitVal[i].size()/validationSize);
		cout<< "numPatchersPerSquare["<<i<<"]: " << numPatchesPerSquare[i] << endl;
	} cout << endl;
	/*
	for(int i=0;i<nMLPs;i++){
	  cout << "size of random feature vector[" <<i<<"]: " << splitTrain[i].size() << endl;
	}*/
	
	trainingSet.clear();
	validationSet.clear();
}

vector<Feature>& MLPController::createRandomFeatureVector(vector<Feature>& trainingData){
    unsigned int nPatches = controller.scannerSettings.nRandomPatches;
   
	vector<Feature> testData;
	cout << "create random feature vector of size: " << nPatches << endl;
	
	for(size_t pIdx = 0; pIdx < nPatches; ++pIdx){
                Patch patch;
                if(controllerSettings.splitTrainSet and controllerSettings.stackSize > 1)
                    patch = imageScanner.getRandomPatch(dataset->getTrainImagePtr(rand() % trainSizeBottomLevel));
                else 
                    patch = imageScanner.getRandomPatch(dataset->getTrainImagePtr(rand() % trainSize));
		Feature newFeat = featExtr.extract(patch);
		newFeat.setSquareId(calculateSquareOfPatch(patch));
		trainingData.push_back(newFeat);    
	}
	
	return trainingData;	
}

vector<vector<Feature> > MLPController::createRandomFeatVal(vector<vector<Feature> >& valSet){
        vector<vector<Feature> > randomFeatVal = vector<vector<Feature> >(nMLPs);
        for(int i=0;i < nMLPs;i++){
            double ratio = (double)controller.scannerSettings.nRandomPatches/(double)trainSize;
            double sizeRandomFeatVal = (double)validationSize*ratio;
            for(int j = 0; j < sizeRandomFeatVal; j++)
                randomFeatVal[i].push_back(valSet[i][rand() % (numPatchesPerSquare[i]*validationSize)]);
        }
        return randomFeatVal;
}
//--------------------end: bottom level---------------------------
//--------------------start: first level--------------------------

double Rand(double fMin, double fMax){
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

bool isThereANextPoolingType(string type){
    return type.empty() ? 0 : 1;
}

string currentPoolingType(string& type){
    size_t iter = type.find_first_of(",");
    string currentType;
    
    if(iter != string::npos){
        currentType = type.substr(0,iter);
        type = type.substr(iter+1);
    }
    else{
        currentType = type;
        type = "";
    }
    
    return currentType;
}

void MLPController::setFirstLevelData(vector<vector<Feature> >& splitDataBottom,vector<Feature>& dataFirstLevel, int sizeData){
	for(int i = 0; i < sizeData;i++){			
		vector<double> inputVector;
		inputVector.reserve(nHiddenBottomLevel*nMLPs*3);
                string type = controllerSettings.poolingType;
                //In the settings file a string is defined with the pooling types with a comma seperator
                //e.g. AVERAGE,MAX. Are the abbriviations for max pooling and average pooling
                //The pooling types that are available are MAX, MIN, and AVERAGE. 
                while(isThereANextPoolingType(type)){
                    string currentType = currentPoolingType(type);
                    for(int j=0;j<nMLPs;j++){			
                            vector<Feature>::const_iterator first = splitDataBottom[j].begin()+(numPatchesPerSquare[j]*i);
                            vector<Feature>::const_iterator last = splitDataBottom[j].begin()+(numPatchesPerSquare[j]*(i+1));
                            
                            vector<double> hiddenActivationSquare = mlps[0][j].returnHiddenActivationToMethod(vector<Feature>(first,last),currentType);
                            
                            inputVector.insert(inputVector.end(),hiddenActivationSquare.begin(),hiddenActivationSquare.end());
                            
                            //Is this saver? than above?
                            //for(unsigned int i=0;i<hiddenActivationSquare.size();i++)
                            //   inputVector.push_back(hiddenActivationSquare[i]);
                    }
                    
                    Feature newFeat = new Feature(inputVector);	
                    newFeat.setLabelId(splitDataBottom[0][i*numPatchesPerSquare[0]].getLabelId());
                    dataFirstLevel.push_back(newFeat);
                }
	}
}
/* 
 * In the first level the mlp training is now based on complete images. 
 * This is different from the bottom level where it is based on random patches 
 * */
void MLPController::createDataFirstLevel(vector<Feature>& inputTrainFirstLevel, vector<Feature>& inputValFirstLevel, vector<Feature>& testSetFirstLevel){
  	vector<Feature> trainingSet;
	vector<Feature> validationSet;
	vector<Feature> testSet;
        
	//increasing the stride to decrease the size of the complete picture set
	imageScanner.setScannerStride(controllerSettings.scanStrideFirstLayer);
    
        testSet = createTestSet();
        
        if(controllerSettings.splitTrainSet)
            trainingSet = createCompletePictureSet(trainingSet,trainSizeBottomLevel,trainSize);
        else
            trainingSet = createCompletePictureSet(trainingSet,0,trainSize);
        
	validationSet = createCompletePictureSet(validationSet,trainSize,trainSize+validationSize);
	
	normalizeInput(trainingSet,0);
	normalizeInput(validationSet,0);
        normalizeInput(testSet,0);
	
        vector<vector<Feature> > splitTrain = splitUpDataBySquare(trainingSet);
	vector<vector<Feature> > splitVal   = splitUpDataBySquare(validationSet);
        vector<vector<Feature> > testSetSplit = splitUpDataBySquare(testSet); //This is neccesary for the setFirstLevelData function

	//set new number of patches per square
	for(int i=0;i<nMLPs;i++){
		numPatchesPerSquare[i] = splitVal[i].size()/validationSize;
		cout<< "numPatchersPerSquare with stride increase[" << i << "]: " << numPatchesPerSquare[i] << endl;
	} cout << endl;
	
	trainingSet.clear();
	validationSet.clear();
        testSet.clear();
        
	if(controllerSettings.splitTrainSet)
            setFirstLevelData(splitTrain,inputTrainFirstLevel,trainSizeFirstLevel); 
        else 
            setFirstLevelData(splitTrain,inputTrainFirstLevel,trainSize); 	
        
        setFirstLevelData(splitVal,inputValFirstLevel,validationSize);
        setFirstLevelData(testSetSplit,testSetFirstLevel,dataset->getTestSize());
        
	//cout << "size of input vector first level: " << inputTrainFirstLevel.size() << endl;
	
	setMinAndMaxValueNorm(inputTrainFirstLevel,1); //set min and max value for first level normalization
	
	normalizeInput(inputTrainFirstLevel,1);
	normalizeInput(inputValFirstLevel,1);
        normalizeInput(testSetFirstLevel,1);

	//Change the range of the data from 0,1 to -1,1
	//changeRange(inputTrainFirstLevel,0,0.5);
	//changeRange(inputValFirstLevel,0,0.5);
}

//-----------start: training MLP's-------------------------
void MLPController::trainMutipleMLPs()
{
	vector<vector<Feature> > splitTrain;
	vector<vector<Feature> > splitVal;
                
        vector<vector<Feature> > randomFeatValidation;
	
	if(!controllerSettings.readMLP){
            createDataBottomLevel(splitTrain,splitVal);
            /*
            if(nMLPs == 1){
                vector<Feature> testSet = createTestSet();
                normalizeInput(testSet,0);
                mlps[0][0].train(splitTrain[0],splitVal[0],testSet,numPatchesPerSquare[0]);

            }else*/ 
                for(int i=0;i<nMLPs;i++){ 		
                    mlps[0][i].train(splitTrain[i],splitVal[i],numPatchesPerSquare[i]);
                    cout << "mlp["<<i<<"] from level 0 finished training on randomfeat" << endl << endl;
                }
            
            //Training on validation is done with all patches of an image
            
            //randomFeatValidation = createRandomFeatVal(splitVal);
            
            //cout << "RandomFeat validation set: " << randomFeatValidation[0].size() << endl;
            /*
            for(int i=0;i<nMLPs;i++){
                    mlps[0][i].train(splitVal[i],numPatchesPerSquare[i]);
                    cout << "mlp["<<i<<"] from level 0 finished training on validation set" << endl << endl;
            }*/
            
	} else 
	    importPreTrainedMLP(controllerSettings.readMLPName);
	
        if(controllerSettings.saveMLP)
	  exportTrainedMLP(controllerSettings.saveMLPName);
        
    //cout << "create training data for first level... " << endl;
    if(controllerSettings.stackSize == 2){
        vector<Feature> inputTrainFirstLevel;
        vector<Feature> inputValFirstLevel;
        vector<Feature> testSetFirstLevel;
        
        createDataFirstLevel(inputTrainFirstLevel,inputValFirstLevel,testSetFirstLevel);
    
        //cout << "min value first level: " << minValues[1] << endl;
        //cout << "max value first level: " << maxValues[1] << endl;
    
        /*for(unsigned int i = 0; i < inputTrainFirstLevel.size();i++){
                cout << "feature ["<<i<<"]: " << inputTrainFirstLevel[i].getLabelId() << endl;
                for(int j = 0; j < inputTrainFirstLevel[i].size;j++){
                        cout << inputTrainFirstLevel[i].content[j] << ", ";
                }
                cout << endl << endl;
        }
        */
        mlps[1][0].setEpochs(controllerSettings.epochsSecondLayer);

        //setSettingsSecondLayer();
        //Training on the training set and validating on both the validation and test set
        mlps[1][0].train(inputTrainFirstLevel,inputValFirstLevel,testSetFirstLevel,1); 
        
        //Training on the validation set and validating on the testSet
        //mlps[1][0].train(inputValFirstLevel,testSetFirstLevel,1); 
        //cout << "mlp[0] from level 1 finished training on the validation set" << endl;
        
    }
}
//--------------end: training MLP's-----------------------------
//-------------start: MLP classification------------------------

void MLPController::dropOutTesting(vector<vector<vector<double> > >& newWeights)
{
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
	
	if(controllerSettings.stackSize == 1){
	/*  if(first == 1){
	    vector<vector<vector<double > > > newWeights;
	    
	    for(int i=0;i<nMLPs;i++){
	     newWeights = mlps[0][i].getWeightMatrix();
	     dropOutTesting(newWeights);
	     mlps[0][i].setWeightMatrix(newWeights);
	    }
	    first=0;
	  }*/
		vector<double> votingHistogram = vector<double>(mlpSettings.nOutputUnits,0.0);
		vector<double> outputProp;
		for(int i=0;i<nMLPs;i++){
			outputProp = mlps[0][i].classifyPooling(testFeaturesBySquare[i]);
			for(int j=0;j<mlpSettings.nOutputUnits;j++){
				votingHistogram[j] += outputProp[j];
			}
		}
		double highestProp = 0;
		int mostVotedClass = -1;
		for(int i=0;i<mlpSettings.nOutputUnits;i++){
			if(votingHistogram[i] > highestProp){
				highestProp = votingHistogram[i];
				mostVotedClass = i;
			}
		}
		answer = mostVotedClass;
	}
	
	if(controllerSettings.stackSize == 2){
		vector<Feature> testDataFirstLevel; //empty feature vector that will be filled with first level features
		setFirstLevelData(testFeaturesBySquare,testDataFirstLevel,numOfImages);
		/*
		for(int i=0;i<testDataFirstLevel[0].size;i++){
			cout << testDataFirstLevel[0].content[i] << ", "; 
		}
		cout << endl;
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
		//	cout << votingHisto[i] << endl;
		//}
		//cout << endl;
		
		//cout << "answer: " << answer << endl;
		
		answer = mlps[1][0].classify(testDataFirstLevel);
	}
	return answer;
}
//---------------------end:MLP classification------------------------
//---------------------start:import/export---------------------------
union chardouble{
   char chars[8];
   double doubleVal;
};

union charInt{
   char chars[4];
   unsigned int intVal;
};

void MLPController::importPreTrainedMLP(string filename){
	charInt fancyInt;
	chardouble fancydouble;
	
	ifstream file(filename.c_str(), ios::binary);
	
	file.read(fancyInt.chars,4);
	int readInInputUnits = fancyInt.intVal;
   	if(readInInputUnits != mlpSettings.nInputUnits){
		cout << "The input units that is read in is " << readInInputUnits << " and in the settings file you have " << mlpSettings.nInputUnits << ", please change this" << endl;
		exit(-1);
	}
	
	file.read(fancyInt.chars,4);
	int readInHiddenUnits = fancyInt.intVal;
   	if(readInHiddenUnits != mlpSettings.nHiddenUnits){
		cout << "The hidden units that is read in is " << readInHiddenUnits << " and in the settings file you have " << mlpSettings.nHiddenUnits << ", please change this" << endl;
		exit(-1);
	}
	
	file.read(fancyInt.chars,4);
	int readInOutputUnits = fancyInt.intVal;
   	if(readInOutputUnits != mlpSettings.nOutputUnits){
		cout << "The output units that is read in is " << readInOutputUnits << " and in the settings file you have " << mlpSettings.nOutputUnits << ", please change this" << endl;
		exit(-1);
	}
	
	file.read(fancyInt.chars,4);
	int readInNSplitsForPooling = fancyInt.intVal;
   	if(readInNSplitsForPooling != controllerSettings.nSplitsForPooling){
		cout << "The nSplitsForPooling that is read in is " << readInNSplitsForPooling << " and in the settings file you have " << controllerSettings.nSplitsForPooling << ", please change this" << endl;
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
		cout << "The dataset that is read in is " << readInDatasetType << " and in the settings file you have " << dataset->getType() << ", please change this" << endl;
		exit(-1);
	}
	//read min value first level
	file.read(fancydouble.chars,8);
	minValues[0] = fancydouble.doubleVal;
	
	//read max value first level
	file.read(fancydouble.chars,8);
	maxValues[0] = fancydouble.doubleVal;
	
	int maxUnits=0;
	
	if(maxUnits < mlpSettings.nInputUnits){
	  maxUnits = mlpSettings.nInputUnits;
	}
	if(maxUnits < mlpSettings.nHiddenUnits){
	  maxUnits = mlpSettings.nHiddenUnits;
	}
	if(maxUnits < mlpSettings.nOutputUnits){
	  maxUnits = mlpSettings.nOutputUnits;
	}
	
	vector<vector<double> > biasNodes = vector<vector<double> >(mlpSettings.nLayers-1,vector<double>(maxUnits,0.0));
	vector<vector<vector<double> > > weights = vector<vector<vector<double> > >(mlpSettings.nLayers-1,vector<vector<double> >(maxUnits,vector<double>(maxUnits,0.0)));
	
	for(int i=0; i<nMLPs;i++){ //amount of mlps that needs to be read in
		for(int j=0;j<mlpSettings.nLayers-1;j++){ //amount of weight vectors
			for(int k=0;k<maxUnits;k++){ //amount of collums
				for(int l=0;l<maxUnits;l++){ // amount of rows
				  file.read(fancydouble.chars,8);
				  weights[j][k][l] = fancydouble.doubleVal;
					
				}
			}
		}
		for(int j=0;j<mlpSettings.nLayers-1;j++){ //amount of bais nodes matrixs
			for(int k=0;k<maxUnits;k++){ // max amount of units, there is only one column and 
				file.read(fancydouble.chars,8);
				biasNodes[j][k] = fancydouble.doubleVal;
			}
		}
	//load the weights and bais matrix into the mlp
	mlps[0][i].loadInMLP(weights,biasNodes);
	
	//reset weight matrixes for the next mlp
	biasNodes = vector<vector<double> >(mlpSettings.nLayers-1,vector<double>(maxUnits,0.0));
	weights = vector<vector<vector<double> > >(mlpSettings.nLayers-1,vector<vector<double> >(maxUnits,vector<double>(maxUnits,0.0)));
	
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
	 chardouble fancydouble;
	 
	 ofstream file(filename.c_str(), ios::binary);
	 
	 //write nInputUnits
	 fancyInt.intVal = mlpSettings.nInputUnits;
	 file.write(fancyInt.chars, 4);
	 
	 //write nHiddenUnits
	 fancyInt.intVal = mlpSettings.nHiddenUnits;
	 file.write(fancyInt.chars, 4);
	 
	 //write nOutputUnits
	 fancyInt.intVal = mlpSettings.nOutputUnits;
	 file.write(fancyInt.chars, 4);
	 
	 //write nSplitsForPooling
	 fancyInt.intVal = controllerSettings.nSplitsForPooling;
	 file.write(fancyInt.chars, 4);
	 
	//write dataset used
	switch(controller.datasetSettings.type){
	  case DATASET_CIFAR10:
	    fancyInt.intVal=0;
         break;
	  case DATASET_MNIST:
	    fancyInt.intVal=1;
         break;   
	}
	 file.write(fancyInt.chars, 4);
	
	 //write min value for first layer
	 fancydouble.doubleVal = minValues[0]; 
	file.write(fancydouble.chars, 8);
	
	//write max value for first layer
	fancydouble.doubleVal = maxValues[0];
	file.write(fancydouble.chars, 8);
	 
	for(int i=0; i<nMLPs;i++){ //amount of mlps
		vector<vector<double> > biasNodes = mlps[0][i].getBiasNodes();
		vector<vector<vector<double> > > weights = mlps[0][i].getWeightMatrix();
		
		for(unsigned int j=0;j<weights.size();j++){ //amount of weight vectors, 2 for 3 layer mlp
			for(unsigned int k=0;k<weights[j].size();k++){ //size of the max amount of nodes of the mlp
				for(unsigned int l=0;l<weights[j][k].size();l++){ //size of the max amount of nodes of the mlp
					fancydouble.doubleVal = weights[j][k][l];
					file.write(fancydouble.chars, 8);
				}
			}
		}
		for(unsigned int j=0;j<biasNodes.size();j++){ //amount of bias node vectors
			for(unsigned int k=0;k<biasNodes[j].size();k++){ //size of max amount of nodes of the mlp
				fancydouble.doubleVal = biasNodes[j][k];
				file.write(fancydouble.chars, 8);
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
   chardouble fancydouble;
 
   ofstream file(filename.c_str(),  ios::binary);
   
   //write dataset used
	switch(controller.datasetSettings.type){
      case DATASET_CIFAR10:
         fancyInt.intVal=0;
         break;
      case DATASET_MNIST:
		 fancyInt.intVal=1;
         break;   
	}
   file.write(fancyInt.chars, 4);
 
   //write amount of features
   fancyInt.intVal = controller.scannerSettings.nRandomPatches;   
   file.write(fancyInt.chars, 4);
   
   //write patchWidth
   fancyInt.intVal = controller.scannerSettings.patchWidth;
   file.write(fancyInt.chars, 4);
   
   //write patchWidth
   fancyInt.intVal = controller.scannerSettings.patchHeight;
   file.write(fancyInt.chars, 4);
   
   //write FeatSize
   fancyInt.intVal = mlpSettings.nInputUnits;
   file.write(fancyInt.chars, 4);   
   
   //write FeatExtractor method
   	switch(controller.featureSettings.featureType){
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
			fancydouble.doubleVal = featureVector[i].content[j];
			file.write(fancydouble.chars, 8);
		}
		//write label of the feat
		fancyInt.intVal = featureVector[i].getLabelId();
		//cout << "labels written to the file: " << featureVector[i].getLabelId() << endl
		file.write(fancyInt.chars, 4);
		
		//write pool it came from
		fancyInt.intVal = featureVector[i].getSquareId(); //does this always exists??
		file.write(fancyInt.chars, 4);		
	}		
	if(featureVector.size() == controller.scannerSettings.nRandomPatches){
		//write min value from bottom level
		fancydouble.doubleVal = minValues[0]; 
		file.write(fancydouble.chars, 8);
		//cout << "min value in write: " << fancydouble.doubleVal << endl;
		//write max value
		fancydouble.doubleVal = maxValues[0];
		file.write(fancydouble.chars, 8);
		//cout << "min value in write: " << fancydouble.doubleVal << endl;

		vector<unsigned int> trainImages = dataset->getTrainImageNums();
		for(int i=0;i<dataset->getTrainSize();i++){
			fancyInt.intVal = trainImages[i];
			//cout << trainImages[i] << endl;
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
	chardouble fancydouble;
  
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
		cout << "The read in dataset is unknown, by default it is set to CIFAR10" << endl;

	}
   
	if(readInDatasetType != dataset->getType()){
		cout << "The dataset that is read in is " << readInDatasetType << " and in the settings file you have " << dataset->getType() << ", please change this" << endl;
		exit(-1);
	}
	
	file.read(fancyInt.chars,4);
	unsigned int readInNRandomPatches = fancyInt.intVal;
	if(controller.scannerSettings.nRandomPatches != readInNRandomPatches){
		cout << "The nRandomPatches that is read in is " << readInNRandomPatches << " in the settings file it is " << controller.scannerSettings.nRandomPatches << ", please change this" << endl;
		exit(-1);
	}
  
	file.read(fancyInt.chars,4);
	unsigned int readInPatchWidth = fancyInt.intVal;
	if(controller.scannerSettings.patchWidth != readInPatchWidth){
		cout << "The patchWidth that is read in is " << readInPatchWidth << " in the settings file it is " << controller.scannerSettings.patchWidth << ", please change this" << endl;
		exit(-1);
	}
  
	file.read(fancyInt.chars,4);
	unsigned int readInPatchHeigth = fancyInt.intVal;
	if(controller.scannerSettings.patchHeight != readInPatchHeigth){
		cout << "The patchHeigth that is read in is " << readInPatchHeigth << " in the settings file it is " << controller.scannerSettings.patchHeight << ", please change this" << endl;
		exit(-1);
	}

	file.read(fancyInt.chars,4);
	int readInFeatSize = fancyInt.intVal;
	if(mlpSettings.nInputUnits != readInFeatSize){
		cout << "The feature size that is read in is " << readInFeatSize << " in the settings file it is " << mlpSettings.nInputUnits << ", please change this" << endl;
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
		 cout << "The read in feature extractor is unknown, by default it is set to HOG" << endl;
	}
	
	if(controller.featureSettings.featureType != readInFeatExt){
		cout << "The Feature extractor that is read in is " << readInFeatExt << " in the settings file it is" << controller.featureSettings.featureType << ", please change this" << endl;
		exit(-1);		
	}
	int sizeOfFeatureVector;
	
	if(filename.find("RandomFeat") == 0){
		sizeOfFeatureVector = readInNRandomPatches;
	}
	else {
		sizeOfFeatureVector = dataset->getTrainSize() * mlpSettings.crossValidationSize * amountOfPatchesImage;
	}

	for(int i=0;i<sizeOfFeatureVector;i++){
		vector<double> contentFeat;
		contentFeat.reserve(readInFeatExt);
		
		for(int j=0;j<readInFeatSize;j++){
			file.read(fancydouble.chars, 8);
			contentFeat.push_back(fancydouble.doubleVal);
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
	
	if(!(file.peek() == ifstream::traits_type::eof())){
		//read min value
		file.read(fancydouble.chars, 8);
		minValues[0] = fancydouble.doubleVal;
	
		//read max value
		file.read(fancydouble.chars, 8);
		maxValues[0] = fancydouble.doubleVal;
		
		vector<unsigned int> readInTrainImages;
		for(int i=0;i<dataset->getTrainSize();i++){
			file.read(fancyInt.chars,4);
			//cout << "imageNum["<<i<<"]: " << fancyInt.intVal << endl;
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
