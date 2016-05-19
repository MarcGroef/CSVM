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

	int nHiddenBottomLevel = s.nHiddenUnits;
	
	//init global variables	
	nMLPs = pow(s.nSplitsForPooling,2);
	validationSize = dataset.getTrainSize()*s.crossValidationSize;
	trainSize = dataset.getTrainSize() - validationSize;
	
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

void MLPController::createDataBottomLevel(){
	vector<Feature> trainingSet;
 	vector<Feature> validationSet;
	
	splitTrain = splitUpDataBySquare(createRandomFeatureVector(trainingSet));
	splitVal   = splitUpDataBySquare(createCompletePictureSet(validationSet,trainSize,trainSize+validationSize));
	
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

void MLPController::setInputFirstLevel(vector<vector<Feature> >& trainingSet){
	//Vectors used in this function are accessed globaly
	//TODO:reserve for the inputTrainNeeds to be done outside of this function.
	int nHiddenBottomLevel = settings.mlpSettings.nHiddenUnits;
	for(int i = 0; i < trainSize;i++){			
		
		vector<double> input;
		input.reserve(nHiddenBottomLevel*nMLPs);
	
		for(int j=0;j<nMLPs;j++){			
			vector<Feature>::const_iterator first = trainingSet[j].begin()+(numPatchesPerSquare[j]*i);
			vector<Feature>::const_iterator last = trainingSet[j].begin()+(numPatchesPerSquare[j]*(i+1));
			
			//vector<double>::const_iterator start = input.begin() + (nHiddenBottomLevel*j);
			//vector<double>::const_iterator end = input.begin() + (nHiddenBottomLevel*(j+1));
			
			vector<double> inputTemp = vector<double>(nHiddenBottomLevel*nMLPs,-10.0);
			
			mlps[0][j].returnHiddenActivation(vector<Feature>(first,last),inputTemp);
			input.insert(input.end(),inputTemp.begin(),inputTemp.end());
		}
		Feature newFeat = new Feature(input);	
		newFeat.setLabelId(trainingSet[0][i*numPatchesPerSquare[0]].getLabelId());
		
		inputTrainFirstLevel.push_back(newFeat);
	}
}

void MLPController::setInputFirstLevel(vector<vector<Feature> >& trainingSet, vector<vector<Feature> >& validationSet){
	//Vectors used in this function are accessed globaly
	//TODO:reserve for the inputTrainNeeds to be done outside of this function.
	int nHiddenBottomLevel = settings.mlpSettings.nHiddenUnits;
	for(int i = 0; i < trainSize;i++){			
		
		vector<double> input;
		input.reserve(nHiddenBottomLevel*nMLPs);
	
		for(int j=0;j<nMLPs;j++){			
			vector<Feature>::const_iterator first = trainingSet[j].begin()+(numPatchesPerSquare[j]*i);
			vector<Feature>::const_iterator last = trainingSet[j].begin()+(numPatchesPerSquare[j]*(i+1));

			vector<double> inputTemp = vector<double>(nHiddenBottomLevel,-10.0);
			
			mlps[0][j].returnHiddenActivation(vector<Feature>(first,last),inputTemp);
			input.insert(input.end(),inputTemp.begin(),inputTemp.end());
		}
		Feature newFeat = new Feature(input);	
		newFeat.setLabelId(trainingSet[0][i*numPatchesPerSquare[0]].getLabelId());
		
		inputTrainFirstLevel.push_back(newFeat);
	}
	std::cout << "inputTrainFirstLevel: " << inputTrainFirstLevel.size() << std::endl;
	
	for(int i = 0; i < validationSize;i++){			
		vector<double> input;
		input.reserve(nHiddenBottomLevel*nMLPs);
		
		for(int j=0;j<nMLPs;j++){			
			vector<Feature>::const_iterator first = validationSet[j].begin()+(numPatchesPerSquare[j]*i);
			vector<Feature>::const_iterator last = validationSet[j].begin()+(numPatchesPerSquare[j]*(i+1));

			vector<double> inputTemp = vector<double>(nHiddenBottomLevel,-10.0);
			
			mlps[0][j].returnHiddenActivation(vector<Feature>(first,last),inputTemp);
			input.insert(input.end(),inputTemp.begin(),inputTemp.end());
		}
		
		Feature newFeat = new Feature(input);	
		newFeat.setLabelId(validationSet[0][i*numPatchesPerSquare[0]].getLabelId());
		
		inputValFirstLevel.push_back(newFeat);
	}
	std::cout << "inputValFirstLevel: " << inputValFirstLevel.size() << std::endl;
}
		
void MLPController::createDataFirstLevel(){
  	inputTrainFirstLevel.reserve(settings.mlpSettings.nHiddenUnits*nMLPs);
  	inputValFirstLevel.reserve(settings.mlpSettings.nHiddenUnits*nMLPs);

  	vector<Feature> trainingSet;
	vector<Feature> validationSet;
	
	splitTrain = splitUpDataBySquare(createCompletePictureSet(trainingSet,0,trainSize));
	splitVal   = splitUpDataBySquare(createCompletePictureSet(validationSet,trainSize,trainSize+validationSize));
	
	for(int i=0;i<nMLPs;i++){
		numPatchesPerSquare.push_back(splitVal[i].size()/validationSize);
		std::cout<< "numPatchersPerSquare[" << i << "]: " << numPatchesPerSquare[i] << std::endl;
	}
	trainingSet.clear();
	
	setInputFirstLevel(splitTrain,splitVal);
	
	splitTrain.clear();
	splitVal.clear();	
}
//-----------end: training/validation set------------------
//-----------start: training MLP's-------------------------
void MLPController::trainMLP(MLPerceptron& mlp,vector<Feature>& trainingSet, vector<Feature>& validationSet){
    double crossvalidationSize = dataset.getTrainSize() * settings.mlpSettings.crossValidationSize;
    int numPatchesPerSquare = validationSet.size()/crossvalidationSize; 
    
    mlp.train(trainingSet,validationSet,numPatchesPerSquare);
}

void MLPController::trainMutipleMLPs(){
	createDataBottomLevel();
	for(int i=0;i<nMLPs;i++){ 		
		trainMLP(mlps[0][i],splitTrain[i],splitVal[i]);
		std::cout << "mlp["<<i<<"] from level 0 finished training" << std::endl;
    }
    
    splitTrain.clear();
    splitVal.clear();
    
    std::cout << "create training data for first level... " << std::endl;
    createDataFirstLevel();
    
	trainMLP(mlps[1][0],inputTrainFirstLevel,inputValFirstLevel);
	std::cout << "mlp[0] from level 1 finished training" << std::endl;
	
	inputTrainFirstLevel.clear();
    inputValFirstLevel.clear();
}
//--------------end: training MLP's-----------------------------
//-------------start: MLP classification------------------------

unsigned int MLPController::mlpMultipleClassify(Image* im){
	//TODO: run the image through the first mlp to get the hidden activation
	//TODO: output the second MLP prediction
	
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
		newFeat.setSquareId(patches[patch].getSquare());
		dataFeatures.push_back(newFeat);
	}
	
	
	vector<vector<Feature> > testFeatures = splitUpDataBySquare(dataFeatures); //normalize
	
	setInputFirstLevel(testFeatures);
	
	int answer = mlps[1][0].classify(inputTrainFirstLevel);
	
	inputTrainFirstLevel.clear();
	
	return answer;
}
//---------------------end:MLP classification------------------------
