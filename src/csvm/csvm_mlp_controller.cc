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
    
    validationSize = dataset->getTrainSize()*controllerSettings.crossValidationSize;
    trainSize = dataset->getTrainSize()-validationSize;
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
    for(int i=0;i<nMLPs;i++){
        numPatchesPerSquare.push_back(0);
    }
    mlps.reserve(controllerSettings.stackSize);
    
    mlps = vector<vector<MLPerceptron> >(controllerSettings.stackSize);

    for(int i = 0; i < nMLPs;i++){
            MLPerceptron mlp;
            if(isTrainOnColorImagesUsed()){
                for(size_t j=0; j<2;j++){
                    MLPerceptron colormlp;
                    colormlp.setSettings(mlpSettings);
                    mlps[0].push_back(colormlp);
                }
            }
            mlp.setSettings(mlpSettings);
            mlps[0].push_back(mlp);
    }
    //settings second parameter	
    if(controllerSettings.stackSize == 2){
            MLPerceptron mlp;
            mlp.setSettings(controller.mlpSettingsFirstLevel);
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

bool MLPController::isTrainOnColorImagesUsed(){
    return controller.featureSettings.hogSettings.binmethod == BYCOLOUR and controller.featureSettings.hogSettings.useColourPixel == true and controller.datasetSettings.type == DATASET_CIFAR10 ? 1 : 0;
}
        

void MLPController::setMinAndMaxValueNorm(vector<Feature>& inputFeatures, int index){
	minValues[index] = inputFeatures[0].content[0];
	maxValues[index] = inputFeatures[0].content[0];

	//compute min and max of all the inputs	
	for(unsigned int i = 0; i < inputFeatures.size();i++){
		float possibleMaxValue = *max_element(inputFeatures[i].content.begin(), inputFeatures[i].content.end());
		float possibleMinValue = *min_element(inputFeatures[i].content.begin(), inputFeatures[i].content.end()); 
	
		if(possibleMaxValue > maxValues[index])
			maxValues[index] = possibleMaxValue;
			
		if(possibleMinValue < minValues[index])
			minValues[index] = possibleMinValue;
	}
}

void MLPController::normalizeInput(vector<Feature>& inputFeatures, int index){
	if (maxValues[index] - minValues[index] != 0)
		for(unsigned int i = 0; i < inputFeatures.size();i++)
			for(int j = 0; j < inputFeatures[i].size;j++)
			  inputFeatures[i].content[j] = (inputFeatures[i].content[j] - minValues[index])/(maxValues[index] - minValues[index]);			
	else
		for(unsigned int i = 0; i<inputFeatures.size();i++)
			for(int j = 0; j < inputFeatures[i].size;j++)
				inputFeatures[i].content[j] = 0;
}

void MLPController::changeRange(vector<Feature>& data, float newMin, float newMax){
	float max = 0.0;
	float min = 2.0;
	
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
			testSet.push_back(newFeat);
		}
	}
	return testSet;
}

vector<Feature> MLPController::createValidationSet(){
	vector<Patch> patches;
    vector<Feature> validationSet;
    int start = trainSize;
    int end = trainSize+validationSize;

	for(int i = start; i < end;i++){
		Image* im = dataset->getTrainImagePtr(i);
		 
		//extract patches
		patches = imageScanner.scanImage(im);
      
		//extract features from all patches
		for(size_t patch = 0; patch < patches.size(); ++patch){
			Feature newFeat = featExtr.extract(patches[patch]);
			newFeat.setSquareId(patches[patch].getSquareId());
			validationSet.push_back(newFeat);
		}
	}

    normalizeInput(validationSet,0);
	return validationSet;
}

vector<vector<Feature> > MLPController::splitUpDataBySquare(vector<Feature> featureSet){
	vector<vector<Feature> > splitBySquares = vector<vector<Feature> >(nMLPs);
	if(nMLPs==1)
         for(unsigned int i = 0;i < featureSet.size();i++)
            splitBySquares[0].push_back(featureSet[i]); 
    if(nMLPs==4)  
        for(unsigned int i = 0;i < featureSet.size();i++)
    		splitBySquares[featureSet[i].getSquareId()].push_back(featureSet[i]);	
	return splitBySquares;
}

//-------------------end: data creation methods---------------
//-------------------start: bottomlevel-----------------------
void MLPController::createDataBottomLevel(vector<vector<vector<Feature> > >& bottomLevelData,vector<string> setTypes){
	if(setTypes[0] == "train" && setTypes[1] == "validation"){
        vector<vector<Feature> > splitTrain = splitUpDataBySquare(createRandomFeatureVector());
        vector<vector<Feature> > splitVal = splitUpDataBySquare(createValidationSet());
        bottomLevelData.push_back(splitTrain);
        bottomLevelData.push_back(splitVal);

        for(int i=0;i<nMLPs;i++){
            numPatchesPerSquare[i]=splitVal[i].size()/validationSize;
            cout << "readNumPatches: " << splitVal[i].size()/validationSize << endl;
        }
     }
    //calcNumPatchesPerSquare();
}

vector<Feature> MLPController::createRandomFeatureVector(){
    unsigned int nPatches = controller.scannerSettings.nRandomPatches;
   
	vector<Feature> testData;
    vector<Feature> trainingData;
	cout << "create random feature vector of size: " << nPatches << endl;
	for(size_t pIdx = 0; pIdx < nPatches; ++pIdx){
                Patch patch;
                if(controllerSettings.splitTrainSet and controllerSettings.stackSize > 1)
                    patch = imageScanner.getRandomPatch(dataset->getTrainImagePtr(rand() % trainSizeBottomLevel));
                else 
                    patch = imageScanner.getRandomPatch(dataset->getTrainImagePtr(rand() % trainSize));
		Feature newFeat = featExtr.extract(patch);
		newFeat.setSquareId(patch.getSquareId());
		trainingData.push_back(newFeat);    
	}
	setMinAndMaxValueNorm(trainingData,0);
    normalizeInput(trainingData,0);

	return trainingData;	
}
/*
vector<vector<Feature> > MLPController::createRandomFeatVal(vector<vector<Feature> >& valSet){
        vector<vector<Feature> > randomFeatVal = vector<vector<Feature> >(nMLPs);
        for(int i=0;i < nMLPs;i++){
            float ratio = (float)controller.scannerSettings.nRandomPatches/(float)trainSize;
            float sizeRandomFeatVal = (float)validationSize*ratio;
            for(int j = 0; j < sizeRandomFeatVal; j++)
                randomFeatVal[i].push_back(valSet[i][rand() % (numPatchesPerSquare[i]*validationSize)]);
        }
        return randomFeatVal;
}*/
//--------------------end: bottom level---------------------------
//--------------------start: first level--------------------------
float Rand(float fMin, float fMax){
    float f = (float)rand() / RAND_MAX;
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

vector<Feature> MLPController::extractHiddenActivation(vector<vector<Feature> > splitDataBottom){
    vector<float> inputVector;
    string type = controllerSettings.poolingType;
    vector<Feature> dataFirstLevel;
    //In the settings file a string is defined with the pooling types with a comma seperator
    //e.g. AVERAGE,MAX. Are the abbriviations for max pooling and average pooling
    //The pooling types that are available are MAX, MIN, and AVERAGE. 
    while(isThereANextPoolingType(type)){
        int dataItr = -1;
        string currentType = currentPoolingType(type);
        for(size_t j=0;j<mlps[0].size();j++){
            isTrainOnColorImagesUsed() ? dataItr = j/3:dataItr = j;
            vector<Feature>::const_iterator first = splitDataBottom[dataItr].begin();
            vector<Feature>::const_iterator last = splitDataBottom[dataItr].end();

            vector<float> hiddenActivationSquare = mlps[0][j].returnHiddenActivationToMethod(vector<Feature>(first,last),currentType);
            
            inputVector.insert(inputVector.end(),hiddenActivationSquare.begin(),hiddenActivationSquare.end());
        }
    }
    Feature newFeat = new Feature(inputVector); 
    newFeat.setLabelId(splitDataBottom[0][0].getLabelId());
    dataFirstLevel.push_back(newFeat);
    return dataFirstLevel;
}

vector<Feature> MLPController::imageToFeatures(string setType, int imageNum){
    vector<Feature> features;
    vector<Patch> patches;
    Image* im = dataset->getTrainImagePtr(imageNum);
    
    if(setType == "test")
        im = dataset->getTestImagePtr(imageNum);
    
    //extract patches
    patches = imageScanner.scanImage(im);
    
    //extract features from all patches
    for(size_t patch = 0; patch < patches.size(); ++patch){
            Feature newFeat = featExtr.extract(patches[patch]);
            newFeat.setSquareId(patches[patch].getSquareId());
            features.push_back(newFeat);
    }
    normalizeInput(features,0);

    return features;
}
/* 
 * In the first level the mlp training is now based on complete images. 
 * This is different from the bottom level where it is based on random patches 
 * */

void MLPController::calcNumPatchesPerSquare(){
    int scannerStride = imageScanner.getScannerStride();

    int patchWidth = controller.scannerSettings.patchWidth;
    int patchHeight = controller.scannerSettings.patchHeight;
    
    if(nMLPs==1)
        numPatchesPerSquare[0] = (((controller.datasetSettings.imWidth - patchWidth)/scannerStride)+1) * (((controller.datasetSettings.imHeight - patchHeight)/scannerStride)+1);
    
    if(nMLPs == 4){
       bool trueMiddelX = 0;
       bool trueMiddelY = 0;

       int maxOffSetWidth = controller.datasetSettings.imWidth-patchWidth;
       int middelOffSetWidth = maxOffSetWidth/2;

       int maxOffSetHeigth = controller.datasetSettings.imHeight-patchHeight;
       int middelOffSetHeigth = maxOffSetHeigth/2;

       if(maxOffSetWidth%2==0)
          trueMiddelX=1;
       if(maxOffSetHeigth%2==0)
          trueMiddelY=1;
      
       if(trueMiddelX && trueMiddelY){
            numPatchesPerSquare[0] = (((middelOffSetWidth)/scannerStride)+1) * (middelOffSetHeigth/scannerStride);
            numPatchesPerSquare[1] = (middelOffSetWidth/scannerStride) * (middelOffSetHeigth/scannerStride);
            numPatchesPerSquare[2] = (((middelOffSetWidth)/scannerStride)+1) * (((middelOffSetHeigth)/scannerStride)+1);
            numPatchesPerSquare[3] = (middelOffSetWidth/scannerStride) * (((middelOffSetHeigth)/scannerStride)+1);
       }
       if(!trueMiddelX && !trueMiddelY)
        for(int i=0;i<nMLPs;i++)
            numPatchesPerSquare[i] = ((middelOffSetWidth/scannerStride)+1) * (((middelOffSetHeigth)/scannerStride)+1);
       //if(trueMiddelX && !trueMiddelY) implement this
        for(int i=0;i<nMLPs;i++)
            cout<< "numPatchersPerSquare with stride " << scannerStride << ", [" <<i<< "]: " << numPatchesPerSquare[i] << endl;
    }
}
void MLPController::createDataFirstLevel(vector<vector<Feature> >& trainingData, vector<string> setTypes){
    imageScanner.setScannerStride(controllerSettings.scanStrideFirstLayer);
    for(size_t i=0;i<setTypes.size();i++){
        unsigned int end=0;
        unsigned int start=0;
        
        if(setTypes[i] == "train")
            end = trainSize;
        if(setTypes[i] == "validation"){
            end = trainSize + validationSize;
            start = trainSize;
        }
        if(setTypes[i] == "test")
            end = dataset->getTestSize();

        for(size_t j=start;j<end;j++){
            vector<vector<Feature> > split = splitUpDataBySquare(imageToFeatures(setTypes[i],j));
            if(j==0 && setTypes[i] == "train")
                for(int k=0;k<nMLPs;k++){
                    numPatchesPerSquare[k] = split[k].size();
                    cout << "readNumPatches: " << split[k].size() << endl;
                }
            vector<Feature> hiddenAcFeatues = extractHiddenActivation(split);
            for(size_t k=0;k<hiddenAcFeatues.size();k++){
                trainingData[i].push_back(hiddenAcFeatues[k]);
             }
        }
        if(setTypes[i] == "train")
            setMinAndMaxValueNorm(trainingData[i],1);
        normalizeInput(trainingData[i],1);
     }
}

vector<Feature> MLPController::splitDataForOneMLPPerColor(vector<Feature>& features, int j){
    vector<Feature> partialFeatureVec;
    int inputSizeOneColorMLP = controller.mlpSettings.nInputUnits;

    for(size_t k=0;k<features.size();k++){
        vector<float> inputColor = vector<float>(features[k].content.begin() + inputSizeOneColorMLP*j,features[k].content.begin() + inputSizeOneColorMLP*(j+1));
        
        Feature tempFeat = new Feature(inputColor);
        tempFeat.setLabelId(features[k].getLabelId());
        partialFeatureVec.push_back(tempFeat);
    }      
    return partialFeatureVec;                     

}
//-----------start: training MLP's-------------------------
void MLPController::trainMutipleMLPs()
{
	//vector<vector<Feature> > splitTrain;
	//vector<vector<Feature> > splitVal;
    vector<string> setTypes;
    vector<vector<vector<Feature> > > bottomLevelData;
    //bottomLevelData = vector<vector<vector<Feature> > >(2); //2 for training on a random feature vector and validation set

	if(!controllerSettings.readMLP){
        setTypes.push_back("train");
        setTypes.push_back("validation");
        
        createDataBottomLevel(bottomLevelData,setTypes);
        if(isTrainOnColorImagesUsed()){
            vector<vector<Feature> > partialData; 
            partialData = vector<vector<Feature> >(2); 
            for(int i=0;i<nMLPs;i++)
                for(int j=i*3;j<3+(i*3);j++){   
                    for(int k=0;k<2;k++)
                        partialData[k] = splitDataForOneMLPPerColor(bottomLevelData[k][i],j-(i*3));
                    mlps[0][j].setNumPatchesPerSquare(numPatchesPerSquare[i]);
                    mlps[0][j].train(partialData[0],partialData[1]);
                }
            }
        else{
            for(int i=0;i<nMLPs;i++){
                mlps[0][i].setNumPatchesPerSquare(numPatchesPerSquare[i]);
                mlps[0][i].train(bottomLevelData[0][i],bottomLevelData[1][i]);
                cout << "mlp["<<i<<"] from level 0 finished training on randomfeat" << endl;
            }
        }
        setTypes.clear();
	} else{
        cout << "loading in mlp..." << endl;
	    importPreTrainedMLP(controllerSettings.readMLPName);
        }
	
        if(controllerSettings.saveMLP)
	       exportTrainedMLP(controllerSettings.saveMLPName);
        
    if(controllerSettings.stackSize == 2){
        vector<vector<Feature> > firstLevelData;        
        firstLevelData = vector<vector<Feature> >(2);

        setTypes.push_back("train");
        setTypes.push_back("validation");

        createDataFirstLevel(firstLevelData,setTypes);

        cout << "lekker" << endl;

        firstLevelData[0].insert(firstLevelData[0].end(),firstLevelData[1].begin(),firstLevelData[1].end());

        mlps[1][0].setNumPatchesPerSquare(1);
        mlps[1][0].train(firstLevelData[0]); 
        cout << "mlp[0] from level 1 finished training on the training set" << endl;   
    }
}
//--------------end: training MLP's-----------------------------
//-------------start: MLP classification------------------------

void MLPController::dropOutTesting(vector<vector<vector<float> > >& newWeights)
{
  float p = 0.5;
  for(unsigned int i=0;i<newWeights.size();i++){
    for(unsigned int  j=0;j<newWeights[i].size();j++){
      for(unsigned int k=0;k<newWeights[i][j].size();k++){
	 newWeights[i][j][k] *= p;
      }
    }
  }
} 
void MLPController::activationsToOutputProbabilities(vector<float>& votingHistogram){
	float sumOfActivations = 0;
	for(int i = 0; i< mlpSettings.nOutputUnits; i++){
		votingHistogram[i] = exp(votingHistogram[i]);
		sumOfActivations += votingHistogram[i];
	}
	for(int i = 0; i< mlpSettings.nOutputUnits; i++)
            votingHistogram[i] /= sumOfActivations;
}
unsigned int MLPController::mlpMultipleClassify(Image* im){
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
		newFeat.setSquareId(patches[patch].getSquareId());
		dataFeatures.push_back(newFeat);
	}
	normalizeInput(dataFeatures,0);
	vector<vector<Feature> > testFeaturesBySquare = splitUpDataBySquare(dataFeatures); //split test features by square	
	
	if(controllerSettings.stackSize == 1){
		vector<float> votingHistogram = vector<float>(mlpSettings.nOutputUnits,0.0);
		vector<float> outputProp;
        if(isTrainOnColorImagesUsed()){
            for(int i=0;i<nMLPs;i++){
                  for(int j=i*3;j<3+(i*3);j++){
                        vector<Feature> partialTest = splitDataForOneMLPPerColor(testFeaturesBySquare[i],j-(i*3));

                        outputProp = mlps[0][j].classifyPooling(partialTest);
                        
                        for(int k=0;k<mlpSettings.nOutputUnits;k++){
                            votingHistogram[k] += outputProp[k];
                        }
                    }
            }
        }
        else{
    		for(int i=0;i<nMLPs;i++){
    			outputProp = mlps[0][i].classifyPooling(testFeaturesBySquare[i]);
    			for(int j=0;j<mlpSettings.nOutputUnits;j++){
    				votingHistogram[j] += outputProp[j];
    			}
    		}
        }
		//activationsToOutputProbabilities(votingHistogram);
		
        float highestProp = 0;
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
		testDataFirstLevel = extractHiddenActivation(testFeaturesBySquare);

		normalizeInput(testDataFirstLevel,1); 

		answer = mlps[1][0].classify(testDataFirstLevel);
	}
	return answer;
}
//---------------------end:MLP classification------------------------
//---------------------start:import/export---------------------------
union charfloat{
   char chars[4];
   float floatVal;
};

union charInt{
   char chars[4];
   unsigned int intVal;
};

void MLPController::importPreTrainedMLP(string filename){
	charInt fancyInt;
	charfloat fancyFloat;
	
	ifstream file(filename.c_str(), ios::binary);
	
        if(!file.is_open()){
            cout << "Cannot find this mlp: "<< filename << endl;
            exit(-1);
        }
        
	file.read(fancyInt.chars,4);
	int readInInputUnits = fancyInt.intVal;
   	if(readInInputUnits != mlpSettings.nInputUnits){
		cout << "The input units that are read in from the mlp is " << readInInputUnits << " and in the settings file you have " << mlpSettings.nInputUnits << ", please change this" << endl;
		exit(-1);
	}
	
	file.read(fancyInt.chars,4);
	int readInHiddenUnits = fancyInt.intVal;
   	if(readInHiddenUnits != mlpSettings.nHiddenUnits){
		cout << "The hidden units that are read in from the mlp is " << readInHiddenUnits << " and in the settings file you have " << mlpSettings.nHiddenUnits << ", please change this" << endl;
		exit(-1);
	}
	
	file.read(fancyInt.chars,4);
	int readInOutputUnits = fancyInt.intVal;
   	if(readInOutputUnits != mlpSettings.nOutputUnits){
		cout << "The output units that are read in from the mlp is " << readInOutputUnits << " and in the settings file you have " << mlpSettings.nOutputUnits << ", please change this" << endl;
		exit(-1);
	}
	
	file.read(fancyInt.chars,4);
	int readInNSplitsForPooling = fancyInt.intVal;
   	if(readInNSplitsForPooling != controllerSettings.nSplitsForPooling){
		cout << "The nSplitsForPooling that are read in from the mlp is " << readInNSplitsForPooling << " and in the settings file you have " << controllerSettings.nSplitsForPooling << ", please change this" << endl;
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
		cout << "The dataset that are read in from the mlp is " << readInDatasetType << " and in the settings file you have " << dataset->getType() << ", please change this" << endl;
		exit(-1);
	}
	//read min value first level
	file.read(fancyFloat.chars,4);
	minValues[0] = fancyFloat.floatVal;
	
	//read max value first level
	file.read(fancyFloat.chars,4);
	maxValues[0] = fancyFloat.floatVal;
	
        file.read(fancyInt.chars,4);
	int dropout = fancyInt.intVal;
   	if(dropout != mlpSettings.dropout){
		cout << "The dropout that are read in from the mlp is " << dropout << " and in the settings file you have " << mlpSettings.dropout << ", please change this" << endl;
		exit(-1);
	}
	
        file.read(fancyInt.chars,4);
	int momentum = fancyInt.intVal;
   	if(momentum != mlpSettings.momentum){
		cout << "The momentum that are read in from the mlp is " << momentum << " and in the settings file you have " << mlpSettings.momentum << ", please change this" << endl;
		exit(-1);
	}
        
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
	
	vector<vector<float> > biasNodes = vector<vector<float> >(mlpSettings.nLayers-1,vector<float>(maxUnits,0.0));
	vector<vector<vector<float> > > weights = vector<vector<vector<float> > >(mlpSettings.nLayers-1,vector<vector<float> >(maxUnits,vector<float>(maxUnits,0.0)));
	
	for(size_t i=0; i<mlps[0].size();i++){ //amount of mlps that needs to be read in from mlp
		for(int j=0;j<mlpSettings.nLayers-1;j++){ //amount of weight vectors
			for(int k=0;k<maxUnits;k++){ //amount of collums
				for(int l=0;l<maxUnits;l++){ // amount of rows
				  file.read(fancyFloat.chars,4);
				  weights[j][k][l] = fancyFloat.floatVal;
					
				}
			}
		}
		for(int j=0;j<mlpSettings.nLayers-1;j++){ //amount of bais nodes matrixs
			for(int k=0;k<maxUnits;k++){ // max amount of units, there is only one column and 
				file.read(fancyFloat.chars,4);
				biasNodes[j][k] = fancyFloat.floatVal;
			}
		}
	//load the weights and bais matrix into the mlp
	mlps[0][i].loadInMLP(weights,biasNodes);
	
	//reset weight matrixes for the next mlp
	biasNodes = vector<vector<float> >(mlpSettings.nLayers-1,vector<float>(maxUnits,0.0));
	weights = vector<vector<vector<float> > >(mlpSettings.nLayers-1,vector<vector<float> >(maxUnits,vector<float>(maxUnits,0.0)));
	
	}
        //read in from mlp same training images and test images
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

void MLPController::exportTrainedMLP(string filename){
	/*
	 * First,  nInputUnits : 0 - maxInt (4 bytes)
	 * Second, nHiddenUnits: 0 - maxInt (4 bytes)
	 * Third,  nOutputUnits: 0 - maxInt usually 10 (4 bytes)
	 * Fourth, nSplitsForPooling : 1-2 (4 bytes)
	 * Fifth,  Dataset     : CIFAR10=0, CIFAR10=1 (4 bytes)
	 * 
	 * Save weight and bias matrix.
	 * 
	 * No seperator char is used 
	 * 
    */	 
	 charInt fancyInt;
	 charfloat fancyFloat;
	 
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
	
	 //write min value for bottom layer
	 fancyFloat.floatVal = minValues[0]; 
	file.write(fancyFloat.chars, 4);
	
	//write max value for bottom layer
	fancyFloat.floatVal = maxValues[0];
	file.write(fancyFloat.chars, 4);
        
        //write if dropout is used
	 fancyInt.intVal = mlpSettings.dropout;
	 file.write(fancyInt.chars, 4);
         
        //write if momentum is used
	 fancyInt.intVal = mlpSettings.momentum;
	 file.write(fancyInt.chars, 4);
	 
	for(size_t i=0; i<mlps[0].size();i++){ //amount of mlps
		vector<vector<float> > biasNodes = mlps[0][i].getBiasNodes();
		vector<vector<vector<float> > > weights = mlps[0][i].getWeightMatrix();
		
		for(unsigned int j=0;j<weights.size();j++){ //amount of weight vectors, 2 for 3 layer mlp
			for(unsigned int k=0;k<weights[j].size();k++){ //size of the max amount of nodes of the mlp
				for(unsigned int l=0;l<weights[j][k].size();l++){ //size of the max amount of nodes of the mlp
					fancyFloat.floatVal = weights[j][k][l];
					file.write(fancyFloat.chars, 4);
				}
			}
		}
		for(unsigned int j=0;j<biasNodes.size();j++){ //amount of bias node vectors
			for(unsigned int k=0;k<biasNodes[j].size();k++){ //size of max amount of nodes of the mlp
				fancyFloat.floatVal = biasNodes[j][k];
				file.write(fancyFloat.chars, 4);
			}
		}
	}
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

void MLPController::exportFeatureSet(string filename, vector<Feature>& featureVector){
	/* Featureset file conventions:
    * 
    * first,  Dataset			: 0, CIFAR10 1,CIFAR10   			(4 bytes)
    * second, Amount of features: 0-10.000.000         			(4 bytes)
    * third,  PatchWidth		: 0-36                 			(4 bytes)
    * fourth, PatchHeigth		: 0-36                 			(4 bytes)  
    * fifth,  FeatSize			: 0-10.000             		   	(4 bytes)
    * sixth,  FeatExtractor		: 0,LBP 1,CLEAN 2,HOG 3,MERGE   (4 bytes)  
    * 
    * from now one it will look like this:
    * 	all float values from the feature
    * 	labelId of the feature
    * 	pool it came from
    *  
    *  No seperator characters are used
   */
   
   charInt fancyInt;
   charfloat fancyFloat;
 
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
			fancyFloat.floatVal = featureVector[i].content[j];
			file.write(fancyFloat.chars, 4);
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
		fancyFloat.floatVal = minValues[0]; 
		file.write(fancyFloat.chars, 4);
		//cout << "min value in write: " << fancyFloat.floatVal << endl;
		//write max value
		fancyFloat.floatVal = maxValues[0];
		file.write(fancyFloat.chars, 4);
		//cout << "min value in write: " << fancyFloat.floatVal << endl;

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
	charfloat fancyFloat;
  
	//unsigned int typesize;
	//unsigned int featDims;
	
	ifstream file(filename.c_str(), ios::binary);
        
        if(!file.is_open()){
            cout << "Cannot find this feature set: "<< filename << endl;
        }
	file.read(fancyInt.chars,4);
	unsigned int datasetNum = fancyInt.intVal; //some check that his num is smaller than 2
	CSVMDatasetType readInDatasetType;
   
   	switch(datasetNum){
      case 0:
         readInDatasetType = DATASET_CIFAR10;
         break;
      case 1:
		 readInDatasetType = DATASET_CIFAR10;
		 break;
	   default:
		readInDatasetType = DATASET_CIFAR10;
		cout << "The read in from feature set dataset is unknown, by default it is set to CIFAR10" << endl;

	}
   
	if(readInDatasetType != dataset->getType()){
		cout << "The dataset that are read in from the feature set is " << readInDatasetType << " and in the settings file you have " << dataset->getType() << ", please change this" << endl;
		exit(-1);
	}
	
	file.read(fancyInt.chars,4);
	unsigned int readInNRandomPatches = fancyInt.intVal;
	if(controller.scannerSettings.nRandomPatches != readInNRandomPatches){
		cout << "The nRandomPatches that are read in from the feature set is " << readInNRandomPatches << " in the settings file it is " << controller.scannerSettings.nRandomPatches << ", please change this" << endl;
		exit(-1);
	}
  
	file.read(fancyInt.chars,4);
	unsigned int readInPatchWidth = fancyInt.intVal;
	if(controller.scannerSettings.patchWidth != readInPatchWidth){
		cout << "The patchWidth that are read in from the feature set is " << readInPatchWidth << " in the settings file it is " << controller.scannerSettings.patchWidth << ", please change this" << endl;
		exit(-1);
	}
  
	file.read(fancyInt.chars,4);
	unsigned int readInPatchHeigth = fancyInt.intVal;
	if(controller.scannerSettings.patchHeight != readInPatchHeigth){
		cout << "The patchHeigth that are read in from the feature set is " << readInPatchHeigth << " in the settings file it is " << controller.scannerSettings.patchHeight << ", please change this" << endl;
		exit(-1);
	}

	file.read(fancyInt.chars,4);
	int readInFeatSize = fancyInt.intVal;
	if(mlpSettings.nInputUnits != readInFeatSize){
		cout << "The feature size that are read in from the feature set is " << readInFeatSize << " in the settings file it is " << mlpSettings.nInputUnits << ", please change this" << endl;
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
		sizeOfFeatureVector = dataset->getTrainSize() * controllerSettings.crossValidationSize * amountOfPatchesImage;
	}

	for(int i=0;i<sizeOfFeatureVector;i++){
		vector<float> contentFeat;
		contentFeat.reserve(readInFeatExt);
		
		for(int j=0;j<readInFeatSize;j++){
			file.read(fancyFloat.chars, 4);
			contentFeat.push_back(fancyFloat.floatVal);
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
		file.read(fancyFloat.chars, 4);
		minValues[0] = fancyFloat.floatVal;
	
		//read max value
		file.read(fancyFloat.chars, 4);
		maxValues[0] = fancyFloat.floatVal;
		
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
