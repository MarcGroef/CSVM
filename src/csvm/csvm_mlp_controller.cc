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

void MLPController::setSettings(MLPSettings s){
	mlps.reserve(s.nMLPs);
	for(int i = 0; i < s.nMLPs;i++){
		MLPerceptron mlp;
		std::cout << "mlp["<<i<<"]:";
		mlp.setSettings(s); 
		mlps.push_back(mlp);
	}
} 
 
void MLPController::initMLPs(){
	//mlps.reserve(settings.mlpSettings.nMLPs);
	//for(int i = 0; i < settings.mlpSettings.nMLPs;i++){
	//	MLPerceptron mlp;
	//	std::cout << "mlp["<<i<<"]:";
	//	mlp.setSettings(settings.mlpSettings); 
	//	mlps.push_back(mlp);
	//}
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


vector<Feature>& MLPController::createValidationSet(vector<Feature>& validationSet){
	int amountOfImagesCrossVal = dataset.getTrainSize() * settings.mlpSettings.crossValidationSize;
	int noPatchesPerSquare  = validationSet.size() / amountOfImagesCrossVal; //Needs a fix, is now 0
	
	vector<Patch> patches;   
  
    validationSet.reserve(noPatchesPerSquare*amountOfImagesCrossVal);
  
	for(int i = dataset.getTrainSize() - amountOfImagesCrossVal; i < dataset.getTrainSize();i++){
		Image* im = dataset.getTrainImagePtr(i);
		 
		//extract patches
		patches = imageScanner.scanImage(im);
      
		//extract features from all patches
		for(size_t patch = 0; patch < patches.size(); ++patch){
			Feature newFeat = featExtr.extract(patches[patch]);
			newFeat.setSquareId(patches[patch].getSquare());
			validationSet.push_back(newFeat);
		}
	}
	return validationSet;
}

vector<Feature>& MLPController::createRandomFeatureVector(vector<Feature>& trainingData){
    unsigned int nPatches = settings.scannerSettings.nRandomPatches;
    int sizeTrainingSet = (int)(dataset.getTrainSize()*(1-settings.mlpSettings.crossValidationSize));
    
	vector<Feature> testData;
  
	std::cout << "Feature extraction training set..." << std::endl;
   for(size_t pIdx = 0; pIdx < nPatches; ++pIdx){
      Patch patch = imageScanner.getRandomPatch(dataset.getTrainImagePtr(rand() % sizeTrainingSet));
      //std::cout << "(mlp controller) getSquare: " << patch.getSquare() << std::endl;
      Feature newFeat = featExtr.extract(patch);
      newFeat.setSquareId(patch.getSquare());
      trainingData.push_back(newFeat);    
   }
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
		trainMLP(mlps[i],splitTrain[i],splitVal[i]);
		std::cout << "(classifier) trained mlp["<<i<<"]" << std::endl;
    }
    splitTrain.clear();
    splitVal.clear();
}

vector<vector<Feature> > MLPController::splitUpDataBySquare(vector<Feature>& trainingSet){
	vector<vector<Feature> > splitBySquares = vector<vector<Feature> >(settings.mlpSettings.nMLPs);
			
	for(unsigned int i = 0;i < trainingSet.size();i++){
		splitBySquares[trainingSet[i].getSquareId()].push_back(trainingSet[i]);	
	}
	return splitBySquares;
}


unsigned int MLPController::mlpClassify(Image* im){
	
	vector<Patch> patches;
	vector<Feature> dataFeatures;
      
	//extract patches
    patches = imageScanner.scanImage(im);

    //allocate for new features
    dataFeatures.reserve(patches.size());
      
    //extract features from all patches
    for(size_t patch = 0; patch < patches.size(); ++patch)
		dataFeatures.push_back(featExtr.extract(patches[patch]));
		
	return mlps[0].classify(dataFeatures);
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
		newFeat.setSquareId(patches[patch].getSquare());
		dataFeatures.push_back(newFeat);
	}
		
  vector<vector<Feature> > testFeatures = splitUpDataBySquare(dataFeatures);
	vector<double> oneSquare = vector<double>(settings.mlpSettings.nOutputUnits,0);      
  
  for(unsigned int i=0;i<mlps.size();i++){
		oneSquare = mlps[i].classifyPooling(testFeatures[i]);
		for(int j =0;j<settings.mlpSettings.nOutputUnits;j++){
			votingHistogramAllSquares[j] += oneSquare[j];
		}
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
