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

//----------------Start initialize MLP's--------------------------------
void MLPController::setSettings(MLPSettings s){
	std::cout << "set settings..." << std::endl;
	

	
	//Initialize global variables
	nMLPs = pow(s.nSplitsForPooling,2);
	validationSize = dataset.getTrainSize()*s.crossValidationSize;
	trainSize = dataset.getTrainSize() - validationSize;
	
	//reserve global vectors
	numPatchesPerSquare.reserve(nMLPs);
	mlps.reserve(2); //2 is the number of layers of MLPs
	
	mlps = vector<vector<MLPerceptron> >(nMLPs);

	for(int i = 0; i < nMLPs; i++){
		MLPerceptron mlp;
		mlp.setSettings(s); 
		mlps[0].push_back(mlp);
	}
	
	//settings second parameter	
	MLPerceptron mlp;
	s.nInputUnits = s.nOutputUnits * nMLPs;
	//std::cout << "s.nInputUnits: " << s.nInputUnits << std::endl;
	s.nHiddenUnits = s.nHiddenSecondLayerMLP;//find parameter
	mlp.setSettings(s);
	mlps[1].push_back(mlp);
	
} 
//----------------End initialize MLP's----------------------------------
 
//----------------Start training/validation set-------------------------
 
 
 int MLPController::calculateSquareOfPatch(Patch patch){
	int splits = settings.mlpSettings.nSplitsForPooling;
	
	int imWidth  = settings.datasetSettings.imWidth;
	int imHeight = settings.datasetSettings.imHeight;
	
	int middlePatchX = patch.getX() + patch.getWidth() / 2;
	int middlePatchY = patch.getY() + patch.getHeight() / 2;
	
	int square = middlePatchX / (imWidth/splits) + (splits * (middlePatchY / (imHeight/splits)));
	return square;
}

void MLPController::createDataFirstLayerMLP(){
	vector<Feature> trainingSet;
 	vector<Feature> validationSet;
	
	splitTrain = splitUpDataBySquare(createRandomFeatureVector(trainingSet));
	splitVal   = splitUpDataBySquare(createCompletePictureSet(validationSet,trainSize,trainSize+validationSize));
	
	for(int i=0;i<nMLPs;i++){
		setMinAndMaxValueNorm(splitTrain[i]);
	}
	
	trainingSet.clear();
	validationSet.clear();
}

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

vector<Feature>& MLPController::createRandomFeatureVector(vector<Feature>& trainingData){
    unsigned int nPatches = settings.scannerSettings.nRandomPatches;
    
	vector<Feature> testData;
	std::cout << "create random feature vector of size:  "<< nPatches << std::endl;
	
    for(size_t pIdx = 0; pIdx < nPatches; ++pIdx){	  
      Patch patch = imageScanner.getRandomPatch(dataset.getTrainImagePtr(rand() % trainSize));
      Feature newFeat = featExtr.extract(patch);
      newFeat.setSquareId(calculateSquareOfPatch(patch));
      //setMinAndMaxValueNorm(newFeat);??????????????????????
      trainingData.push_back(newFeat);   
   }
	return trainingData;	
}

vector<vector<Feature> > MLPController::splitUpDataBySquare(vector<Feature>& trainingSet){
	vector<vector<Feature> > splitBySquares = vector<vector<Feature> >(nMLPs);
			
	for(unsigned int i = 0;i < trainingSet.size();i++){
		splitBySquares[trainingSet[i].getSquareId()].push_back(trainingSet[i]);	
	}
	return splitBySquares;
}

void MLPController::createOutputProbabilitiesVectorTrain(vector<vector<Feature> >& trainingSet, vector<vector<Feature> >&  validationSet){
	unsigned int nOutputProbabilities = settings.mlpSettings.nOutputUnits;
	
	for(int i = 0; i < trainSize; i++){
		
		vector<double> input;
		input.reserve(nOutputProbabilities*nMLPs);
		
		for(int j=0; j<nMLPs; j++){
			vector<Feature>::const_iterator first = trainingSet[j].begin()+(numPatchesPerSquare[j]*i);
			vector<Feature>::const_iterator last = trainingSet[j].begin()+(numPatchesPerSquare[j]*(i+1));
	
			//vector<double>::const_iterator start = input.begin() + (nHiddenBottomLevel*j);
			//vector<double>::const_iterator end = input.begin() + (nHiddenBottomLevel*(j+1));
			
			vector<double> inputTemp = vector<double>(nOutputProbabilities,-10.0);
			
			mlps[0][j].returnOutputActivation(vector<Feature>(first,last),inputTemp);
		
			input.insert(input.end(),inputTemp.begin(),inputTemp.end());
		}
		Feature newFeat = new Feature(input);	
		newFeat.setLabelId(trainingSet[0][i*numPatchesPerSquare[0]].getLabelId());
		
		inputTrainSecondLayerMLP.push_back(newFeat);
	}
	std::cout << "inputTrainSecondLayerMLP: " << inputTrainSecondLayerMLP.size() << std::endl;
	
	for(int i = 0; i < validationSize;i++){			
		vector<double> input;
		input.reserve(nOutputProbabilities*nMLPs);
		
		for(int j=0;j<nMLPs;j++){			
			vector<Feature>::const_iterator first = validationSet[j].begin()+(numPatchesPerSquare[j]*i);
			vector<Feature>::const_iterator last = validationSet[j].begin()+(numPatchesPerSquare[j]*(i+1));

			vector<double> inputTemp = vector<double>(nOutputProbabilities,-10.0);
			
			mlps[0][j].returnOutputActivation(vector<Feature>(first,last),inputTemp);
			input.insert(input.end(),inputTemp.begin(),inputTemp.end());
		}
		
		Feature newFeat = new Feature(input);	
		newFeat.setLabelId(validationSet[0][i*numPatchesPerSquare[0]].getLabelId());
		
		inputValSecondLayerMLP.push_back(newFeat);
	}
	std::cout << "inputValSecondLayerMLP: " << inputValSecondLayerMLP.size() << std::endl;
}

void MLPController::createDataSecondLayerMLP(){
  	inputTrainSecondLayerMLP.reserve(settings.mlpSettings.nOutputUnits*nMLPs);
  	inputValSecondLayerMLP.reserve(settings.mlpSettings.nOutputUnits*nMLPs);

  	vector<Feature> trainingSet;
	vector<Feature> validationSet;
	
	splitTrain = splitUpDataBySquare(createCompletePictureSet(trainingSet,0,trainSize));
	splitVal   = splitUpDataBySquare(createCompletePictureSet(validationSet,trainSize,trainSize+validationSize));
	
	for(int i=0;i<nMLPs;i++){
		numPatchesPerSquare.push_back(splitVal[i].size()/validationSize);
		std::cout<< "numPatchersPerSquare[" << i << "]: " << numPatchesPerSquare[i] << std::endl;
	}
	trainingSet.clear();
	
	createOutputProbabilitiesVectorTrain(splitTrain,splitVal);
	
	splitTrain.clear();
	splitVal.clear();	
}

void MLPController:: createOutputProbabilitiesVectorTest(vector<vector<Feature> >& testSet){
	unsigned int nOutputProbabilities = settings.mlpSettings.nOutputUnits;
	
	vector<double> input;
	input.reserve(nOutputProbabilities*nMLPs);
		
	for(int j=0; j<nMLPs; j++){
		vector<Feature>::const_iterator first = testSet[j].begin();
		vector<Feature>::const_iterator last = testSet[j].begin()+numPatchesPerSquare[j];
			
		vector<double> inputTemp = vector<double>(nOutputProbabilities,-10.0);
			
		mlps[0][j].returnOutputActivation(vector<Feature>(first,last),inputTemp);
		input.insert(input.end(),inputTemp.begin(),inputTemp.end());
	}
	Feature newFeat = new Feature(input);	
	newFeat.setLabelId(testSet[0][0].getLabelId());
		
	inputTrainSecondLayerMLP.push_back(newFeat);
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

vector<Feature>& MLPController::normalizeInput(vector<Feature>& inputFeatures){
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
//----------------End training/validation set---------------------------

//----------------start training MLP's----------------------------------
void MLPController::trainFirstLayerMLP(MLPerceptron& mlp,vector<Feature>& trainingSet, vector<Feature>& validationSet){
    double crossvalidationSize = dataset.getTrainSize() * settings.mlpSettings.crossValidationSize;
    int noPatchesPerSquare = validationSet.size()/crossvalidationSize; 
    mlp.trainFirstLayerMLP(trainingSet,validationSet,noPatchesPerSquare);
}

void MLPController::trainSecondLayerMLP(MLPerceptron& mlp,vector<Feature>& trainingSet, vector<Feature>& validationSet){
    double crossvalidationSize = dataset.getTrainSize() * settings.mlpSettings.crossValidationSize;
    int noPatchesPerSquare = validationSet.size()/crossvalidationSize; 
    mlp.trainSecondLayerMLP(trainingSet,validationSet,noPatchesPerSquare);
}

void MLPController::trainMutipleMLPs(){
    createDataFirstLayerMLP();
	
	for(int i=0;i<nMLPs;i++){
		normalizeInput(splitTrain[i]);
		normalizeInput(splitVal[i]);
	}
	
    for(int i=0;i<nMLPs;i++){ 		
		trainFirstLayerMLP(mlps[0][i],splitTrain[i],splitVal[i]);
		std::cout << "mlp["<<i<<"] from first layer MLP finished training" << std::endl;
    }
    splitTrain.clear();
    splitVal.clear();
    
    std::cout << "create input data for second layer MLP... " << std::endl;
    createDataSecondLayerMLP();
    
    trainSecondLayerMLP(mlps[1][0],inputTrainSecondLayerMLP,inputValSecondLayerMLP);
    std::cout << "mlp[0] from second layer MLP finished training" << std::endl;
	
    inputTrainSecondLayerMLP.clear();
    inputValSecondLayerMLP.clear();
    
}

//---------------end training MLP's-------------------------------------

//---------------Start MLP Classification-------------------------------
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
	
	createOutputProbabilitiesVectorTest(testFeatures);    
	 
	int answer = mlps[1][0].classify(inputTrainSecondLayerMLP);
	
	inputTrainSecondLayerMLP.clear();
	
 	return answer;
}
