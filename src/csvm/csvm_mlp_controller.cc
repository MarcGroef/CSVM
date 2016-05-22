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

MLPController::MLPController(){
	std::cout<<"lekker"<<std::endl;	
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
	s.nInputUnits = nHiddenBottomLevel * 4;
	//std::cout << "s.nInputUnits: " << s.nInputUnits << std::endl;
	s.nHiddenUnits = nHiddenBottomLevel * 5;//find parameter
	mlp.setSettings(s);
	mlps[1].push_back(mlp);
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

//-------------------end: init MLP's-------------------------------------
//-------------------start: training/validation set-----------------------
int MLPController::calculateSquareOfPatch(Patch patch){
	int splits = settings.mlpSettings.nSplitsForPooling;
	
	int middlePatchX = patch.getX() + patch.getWidth() / 2;
	int middlePatchY = patch.getY() + patch.getHeight() / 2;
	
	int imWidth  = settings.datasetSettings.imWidth;
	int imHeight = settings.datasetSettings.imHeight;
	
	return middlePatchX / (imWidth/splits) + (splits * (middlePatchY / (imHeight/splits)));
}

void MLPController::createDataBottomLevel(vector<vector<Feature> >& splitTrain, vector<vector<Feature> >& splitVal){
	vector<Feature> trainingSet;
 	vector<Feature> validationSet;
	
	splitTrain = splitUpDataBySquare(createRandomFeatureVector(trainingSet));
	splitVal   = splitUpDataBySquare(createCompletePictureSet(validationSet,trainSize,trainSize+validationSize));
	
	for(int i = 0; i <nMLPs;i++){
		setMinAndMaxValueNorm(splitTrain[i],0);
	}
	//set number of patches per square.
	for(int i=0;i<nMLPs;i++){
		numPatchesPerSquare.push_back(splitVal[i].size()/validationSize);
		std::cout<< "numPatchersPerSquare[" << i << "]: " << numPatchesPerSquare[i] << std::endl;
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
	std::cout << "create random feature vector of size: " << nPatches << std::endl;
	
	for(size_t pIdx = 0; pIdx < nPatches; ++pIdx){
		Patch patch = imageScanner.getRandomPatch(dataset.getTrainImagePtr(rand() % trainSize));
		Feature newFeat = featExtr.extract(patch);
		newFeat.setSquareId(calculateSquareOfPatch(patch));
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

void MLPController::setInputTestingDataFirstLevel(vector<vector<Feature> >& testFeaturesBySquare, vector<Feature>& testDataFirstLevel){
	int nHiddenBottomLevel = settings.mlpSettings.nHiddenUnits;		
	vector<double> input;
	input.reserve(nHiddenBottomLevel*nMLPs);

	for(int i=0;i<nMLPs;i++){			
		//vector<Feature>::const_iterator first = testFeaturesBySquare[i].begin()+(numPatchesPerSquare[i]);
		//vector<Feature>::const_iterator last = testFeaturesBySquare[i].begin()+(numPatchesPerSquare[i]);
		
		vector<double> inputTemp = vector<double>(nHiddenBottomLevel*nMLPs,-10.0);
		
		mlps[0][i].returnHiddenActivation(testFeaturesBySquare[i],inputTemp);
		input.insert(input.end(),inputTemp.begin(),inputTemp.end());
	}
	Feature newFeat = new Feature(input);	
	//newFeat.setLabelId(trainingSet[0][i*numPatchesPerSquare[0]].getLabelId());
	
	testDataFirstLevel.push_back(newFeat);
}

void MLPController::setFirstLevelTrainData(vector<vector<Feature> >& splitTrain,vector<Feature>& inputTrainFirstLevel){
	for(int i = 0; i < trainSize;i++){			
		vector<double> input;
		input.reserve(nHiddenBottomLevel*nMLPs);
	
		for(int j=0;j<nMLPs;j++){			
			vector<Feature>::const_iterator first = splitTrain[j].begin()+(numPatchesPerSquare[j]*i);
			vector<Feature>::const_iterator last = splitTrain[j].begin()+(numPatchesPerSquare[j]*(i+1));

			vector<double> inputTemp = vector<double>(nHiddenBottomLevel,-10.0);
			
			mlps[0][j].returnHiddenActivation(vector<Feature>(first,last),inputTemp);
			input.insert(input.end(),inputTemp.begin(),inputTemp.end());
		}
		Feature newFeat = new Feature(input);	
		newFeat.setLabelId(splitTrain[0][i*numPatchesPerSquare[0]].getLabelId());
		//std::cout << "input.size[" << i<< "]: " << input.size() << std::endl;
		inputTrainFirstLevel.push_back(newFeat);
		input.clear();
	}
	std::cout << "inputTrainFirstLevel: " << inputTrainFirstLevel.size() << std::endl;
	
}

void MLPController::setFirstLevelValData(vector<vector<Feature> >& splitVal,vector<Feature>& inputValFirstLevel){
		for(int i = 0; i < validationSize;i++){			
		vector<double> input;
		input.reserve(nHiddenBottomLevel*nMLPs);
		
		for(int j=0;j<nMLPs;j++){			
			vector<Feature>::const_iterator first = splitVal[j].begin()+(numPatchesPerSquare[j]*i);
			vector<Feature>::const_iterator last = splitVal[j].begin()+(numPatchesPerSquare[j]*(i+1));

			vector<double> inputTemp = vector<double>(nHiddenBottomLevel,-10.0);
			
			mlps[0][j].returnHiddenActivation(vector<Feature>(first,last),inputTemp);
			input.insert(input.end(),inputTemp.begin(),inputTemp.end());
		}
		
		Feature newFeat = new Feature(input);	
		newFeat.setLabelId(splitVal[0][i*numPatchesPerSquare[0]].getLabelId());
		
		inputValFirstLevel.push_back(newFeat);
	}
	std::cout << "inputValFirstLevel: " << inputValFirstLevel.size() << std::endl;
}

void MLPController::createDataFirstLevel(vector<Feature>& inputTrainFirstLevel, vector<Feature>& inputValFirstLevel){
  	vector<Feature> trainingSet;
	vector<Feature> validationSet;
	
	vector<vector<Feature> > splitTrain = splitUpDataBySquare(createCompletePictureSet(trainingSet,0,trainSize));
	vector<vector<Feature> > splitVal   = splitUpDataBySquare(createCompletePictureSet(validationSet,trainSize,trainSize+validationSize));

	trainingSet.clear();
	validationSet.clear();
	
	for(int i=0;i<nMLPs;i++){
		normalizeInput(splitTrain[i],0);
		normalizeInput(splitVal[i],0);
	}
	
	setFirstLevelTrainData(splitTrain,inputTrainFirstLevel);
	setFirstLevelValData(splitVal,inputValFirstLevel);
    
    std::cout << "input.size(): "<< inputTrainFirstLevel.size() << std::endl;
    
	setMinAndMaxValueNorm(inputTrainFirstLevel,1);
}
//-----------end: training/validation set------------------
//-----------start: training MLP's-------------------------
void MLPController::trainMLP(MLPerceptron& mlp,vector<Feature>& trainingSet, vector<Feature>& validationSet){
    double crossvalidationSize = dataset.getTrainSize() * settings.mlpSettings.crossValidationSize;
    int numPatchesPerSquare = validationSet.size()/crossvalidationSize; 
    
    mlp.train(trainingSet,validationSet,numPatchesPerSquare);
}

void MLPController::trainMutipleMLPs(){
	vector<vector<Feature> > splitTrain;
	vector<vector<Feature> > splitVal;
	
	createDataBottomLevel(splitTrain,splitVal);
	
	for(int i=0;i<nMLPs;i++){
		normalizeInput(splitTrain[i],0);
		normalizeInput(splitVal[i],0);
	}
	
	for(int i=0;i<nMLPs;i++){ 		
		trainMLP(mlps[0][i],splitTrain[i],splitVal[i]);
		std::cout << "mlp["<<i<<"] from level 0 finished training" << std::endl;
    }
    
    std::cout << "create training data for first level... " << std::endl;
    vector<Feature> inputTrainFirstLevel;
	vector<Feature> inputValFirstLevel;
    
    createDataFirstLevel(inputTrainFirstLevel,inputValFirstLevel);
    
    normalizeInput(inputTrainFirstLevel,1);
    normalizeInput(inputValFirstLevel,1);
    
	trainMLP(mlps[1][0],inputTrainFirstLevel,inputValFirstLevel);
	std::cout << "mlp[0] from level 1 finished training" << std::endl;
}
//--------------end: training MLP's-----------------------------
//-------------start: MLP classification------------------------

unsigned int MLPController::mlpMultipleClassify(Image* im){
	vector<double> votingHistogramAllSquares = vector<double>(settings.mlpSettings.nOutputUnits,0);      

	vector<Patch> patches;
	vector<Feature> dataFeatures;

	//extract patches
	patches = imageScanner.scanImage(im);

	//allocate for new features
	dataFeatures.reserve(patches.size());
      
	//extract features from all patches
	for(size_t patch = 0; patch < patches.size(); ++patch){
		Feature newFeat = featExtr.extract(patches[patch]);
		newFeat.setSquareId(patches[patch].getSquare());
		dataFeatures.push_back(newFeat);
	}
	
	vector<vector<Feature> > testFeatures = splitUpDataBySquare(normalizeInput(dataFeatures,0));
	
	vector<Feature> testDataFirstLevel;
	
	setInputTestingDataFirstLevel(testFeatures,testDataFirstLevel);
	
	int answer = mlps[1][0].classify(normalizeInput(testDataFirstLevel,1));
	
	return answer;
}
//---------------------end:MLP classification------------------------
