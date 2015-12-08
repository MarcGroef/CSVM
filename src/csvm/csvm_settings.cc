 #include <csvm/csvm_settings.h>

using namespace std;
using namespace csvm;

CSVMSettings::~CSVMSettings(){
  //free(analyserSettings.rbmSettings.layerSizes);
  
  
}

void CSVMSettings::parseConvSVMSettings(ifstream& stream){
   string setting;
   string method;
   stream >> setting;
   if(setting != "learningRate"){
      cout << "csvm::csvm_settings:parseConvSVMSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }else{
      stream >> convSVMSettings.learningRate;
      
   }
   
   stream >> setting;
   if(setting != "nIterations"){
      cout << "csvm::csvm_settings:parseConvSVMSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }else{
      stream >> convSVMSettings.nIter;
   }
   
   stream >> setting;
   if(setting != "initWeight"){
      cout << "csvm::csvm_settings:parseConvSVMSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }else{
      stream >> convSVMSettings.initWeight;
   }
   
   stream >> setting;
   if(setting != "CSVM_C"){
      cout << "csvm::csvm_settings:parseConvSVMSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }else{
      stream >> convSVMSettings.CSVM_C;
   }
   
   stream >> setting;
   if(setting != "nClasses"){
      cout << "csvm::csvm_settings:parseConvSVMSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }else{
      stream >> convSVMSettings.nClasses;
   }
   stream >> setting;
   if(setting != "nCentroids"){
      cout << "csvm::csvm_settings:parseConvSVMSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }else{
      stream >> convSVMSettings.nCentroids;
   }
   
   
   
   
}

void CSVMSettings::parseLinNetSettings(ifstream& stream){

  
  string setting;
  string method;
  
   stream >> setting;
   if(setting != "initWeight"){
      cout << "csvm::csvm_settings:parseLinNetSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }else{
      stream >> netSettings.initWeight;
      
   }
   stream >> setting;
   if(setting != "learningRate"){
      cout << "csvm::csvm_settings:parseLinNetSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }else{
      stream >> netSettings.learningRate;
   }
   

}

void CSVMSettings::parseDatasetSettings(ifstream& stream){

  
  string setting;
  string method;
  stream >> setting;
  if(setting != "method"){
    cout << "csvm::csvm_settings:parseDatasetSettings(): Error! Invalid settingsfile layout. Exitting...\n";
    exit(-1);
  }
  stream >> method;

  if(method == "CIFAR10"){
      datasetSettings.type = DATASET_CIFAR10;
      stream >> setting;
      if(setting != "nImages"){
         cout << "csvm::csvm_settings:parseDatasetSettings(): In CIFAR10 parsing: Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }
      stream >> datasetSettings.nImages;
    
  }
 
}


/*void CSVMSettings::parseClusterAnalserData(ifstream& stream){
  string setting;
  string method;
  stream >> setting;
  if(setting != "method"){
    cout << "csvm::csvm_settings:parseDatasetSettings(): Error! Invalid settingsfile layout. Exitting...\n";
    exit(-1);
  }
  stream >> method; 

  if(method == "RBM"){
    analyserSettings.method = CSVM_RBM;
    
    stream >> setting;
    if(setting == "nLayers"){
      stream >> analyserSettings.rbmSettings.nLayers;
      analyserSettings.rbmSettings.layerSizes = (int*) malloc(analyserSettings.rbmSettings.nLayers*sizeof(int));
      assert(analyserSettings.rbmSettings.layerSizes!=NULL);
            
    }else{
      cout << "csvm::csvm_settings:parseClusterAnalserData(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
    }
    
    stream >> setting;
    if(setting == "layerSizes"){
      //set layerSizes except the size of the first layer. That one should be determined by the feature extractor feature dimensionality
      for(int idx = 1; idx < analyserSettings.rbmSettings.nLayers; ++idx){
	stream >> analyserSettings.rbmSettings.layerSizes[idx];	
      }
    }else{
      cout << "csvm::csvm_settings:parseClusterAnalserData(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
    }
    
    stream >> setting;
    if(setting == "learningRate"){
      stream >> analyserSettings.rbmSettings.learningRate;
    }else{
      cout << "csvm::csvm_settings:parseClusterAnalserData(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
    }
    
    stream >> setting;
    if(setting == "nGibbsSteps"){
      stream >> analyserSettings.rbmSettings.nGibbsSteps;
    }else{
      cout << "csvm::csvm_settings:parseClusterAnalserData(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
    }
      
    
  }
  
}
*/
void CSVMSettings::parseCodebookSettings(ifstream& stream){
  string setting;
  string method;
  stream >> setting;
  if(setting != "method"){
    cout << "csvm::csvm_settings:parseDatasetSettings(): Error! Invalid settingsfile layout. Exitting...\n";
    exit(-1);
  }
  stream >> method;
  if(method == "LVQ"){
    codebookSettings.method = LVQ_Clustering;
    
    stream >> setting;
    if(setting == "nClusters"){
      stream >> codebookSettings.lvqSettings.nClusters;
    }else{
      cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
    }
    
    stream >> setting;
    if(setting == "learningRate"){
      stream >> codebookSettings.lvqSettings.alpha;
    }else{
      cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
    }
    
    
    
    
  }
  
  if(method == "KMEANS"){
    codebookSettings.method = KMeans_Clustering;
    
    stream >> setting;
    if(setting == "nClusters"){
      stream >> codebookSettings.numberVisualWords;
    }else{
      cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
    }
    
    stream >> setting;
    if(setting == "nIterations"){
       stream >> codebookSettings.kmeansSettings.nIter;
    }else{
      cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
    }
    
    stream >> setting;
    if(setting == "SimilarityFunction"){
      stream >> method;
      if(method == "RBF")
         codebookSettings.simFunction = CB_RBF;
      else if (method == "SOFT_ASSIGNMENT")
         codebookSettings.simFunction = SOFT_ASSIGNMENT;
      else
         cout << "Invalid codebook SimilarityFunction: Try RBF or SOFT_ASSIGNMENT\n";
    }else{
      cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
    }
    
    stream >> setting;
    if(setting == "similaritySigma"){
      stream >> codebookSettings.similaritySigma;
    }else{
      cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
    }
    
    stream >> setting;
    if(setting == "useDifferentCodebooksPerClass"){
       stream >> setting;
      codebookSettings.useDifferentCodebooksPerClass = (setting == "true");
      scannerSettings.useDifferentCodebooksPerClass = (setting == "true");
      svmSettings.useDifferentCodebooksPerClass = (setting == "true");
      datasetSettings.useDifferentCodebooksPerClass = (setting == "true");
      netSettings.useDifferentCodebooksPerClass = (setting == "true");
    }else{
      cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
      
    }
    
    stream >> setting;
    if(setting == "standardizeActivations"){
       stream >> setting;
      codebookSettings.standardizeActivations = (setting == "true");
    }else{
      cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
      
    }
    
  }

  if (method == "AKMEANS") {
	  codebookSettings.method = AKMeans_Clustering;

	  stream >> setting;
	  if (setting == "nClusters") {
		  stream >> codebookSettings.numberVisualWords;
	  }
	  else {
		  cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
		  exit(-1);
	  }

	  stream >> setting;
	  if (setting == "nIterations") {
		  stream >> codebookSettings.akmeansSettings.nIter;

	  }
	  else {
		  cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
		  exit(-1);
	  }

	  stream >> setting;
	  if (setting == "SimilarityFunction") {
		  stream >> method;
		  if (method == "RBF")
			  codebookSettings.simFunction = CB_RBF;
		  else if (method == "SOFT_ASSIGNMENT")
			  codebookSettings.simFunction = SOFT_ASSIGNMENT;
		  else
			  cout << "Invalid codebook SimilarityFunction: Try RBF or SOFT_ASSIGNMENT\n";
	  }
	  else {
		  cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
		  exit(-1);
	  }

	  stream >> setting;
	  if (setting == "similaritySigma") {
		  stream >> codebookSettings.similaritySigma;
	  }
	  else {
		  cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
		  exit(-1);
	  }

	  stream >> setting;
	  if (setting == "useDifferentCodebooksPerClass") {
		  stream >> setting;
		  codebookSettings.useDifferentCodebooksPerClass = (setting == "true");
		  scannerSettings.useDifferentCodebooksPerClass = (setting == "true");
		  svmSettings.useDifferentCodebooksPerClass = (setting == "true");
		  datasetSettings.useDifferentCodebooksPerClass = (setting == "true");
		  netSettings.useDifferentCodebooksPerClass = (setting == "true");
	  }
	  else {
		  cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";

	  }

	  stream >> setting;
	  if (setting == "standardizeActivations") {
		  stream >> setting;
		  codebookSettings.standardizeActivations = (setting == "true");
	  }
	  else {
		  cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";

	  }

  }

}

void CSVMSettings::parseFeatureExtractorSettings(ifstream& stream){
  string setting;
  string method;
  string enumeration;
  stream >> setting;
  if(setting != "method"){
    cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
    exit(-1);
  }
  stream >> method;
  if(method == "LBP"){
    featureSettings.featureType = LBP;  
  }else if(method == "HOG"){
    featureSettings.featureType = HOG;
    
    stream >> setting;
    if(setting == "cellSize"){  // #cellSize is best an even-numbered, divisor of patch size. By default it'll be half of patch size
       stream >> featureSettings.hogSettings.cellSize;
       
    } else {
      cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
    }
    
    stream >> setting;
    if(setting == "cellStride"){ //#cellStride is best an even-numbered, divisor of cellSize. By default it's the same value as cellSize, meaning the patch is divided into quadrants, and not iterated over 
       stream >> featureSettings.hogSettings.cellStride;
       
    } else {
      cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
    }
    
    stream >> setting;
    if(setting == "blockSize"){//#the size of a patch
       stream >> featureSettings.hogSettings.blockSize;
    } else {
      cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
    }
    
    stream >> setting;
    if(setting == "padding"){//#the size of a patch
       stream >> enumeration;
       if(enumeration == "None")
           featureSettings.hogSettings.padding = NONE;
       else if(enumeration == "Identity")
           featureSettings.hogSettings.padding = IDENTITY;
       else if (enumeration == "Zero")
           featureSettings.hogSettings.padding = ZERO;
          
    } else {
      cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
    }
  }else if (method == "CLEAN"){
      featureSettings.featureType = CLEAN;
  }

  
  
}

void CSVMSettings::parseImageScannerSettings(ifstream& stream){
  string setting;
  string method;
  stream >> setting;
  if(setting == "patchHeight"){
    stream >> scannerSettings.patchHeight;   
  }else{
    cout << "csvm::csvm_settings:parseImageScannerSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
    exit(-1);
  }
  
  stream >> setting;
  if(setting == "patchWidth"){
    stream >> scannerSettings.patchWidth; 
  }else{
    cout << "csvm::csvm_settings:parseImageScannerSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
    exit(-1);
  }
  
  stream >> setting;
  if(setting == "scanStride"){
    stream >> scannerSettings.stride;   
  }else{
    cout << "csvm::csvm_settings:parseImageScannerSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
    exit(-1);
  }
  
  stream >> setting;
  if(setting == "nRandomPatches"){
    stream >> scannerSettings.nRandomPatches;   
  }else{
    cout << "csvm::csvm_settings:parseImageScannerSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
    exit(-1);
  }
  
}

void CSVMSettings::parseSVMSettings(ifstream& stream){
  string setting;
  string method;
  
  stream >> setting;
  if(setting == "Type"){
    stream >> method;
    if(method == "CLASSIC")
       svmSettings.type = CLASSIC;
    else if(method == "CONV")
       svmSettings.type = CONV;
    else
       cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
    
  }else{
    cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
    exit(-1);
  }
  
  stream >> setting;
  if(setting == "Kernel"){
    stream >> method;
    if(method == "RBF")
       svmSettings.kernelType = RBF;
    else if(method == "LINEAR")
       svmSettings.kernelType = LINEAR;
    else
       cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
    
  }else{
    cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
    exit(-1);
  }
  
  stream >> setting;
  if(setting == "AlphaDataInit"){
    stream >> svmSettings.alphaDataInit; 
    
  }else{
    cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
    exit(-1);
  }
  
  stream >> setting;
  if(setting == "AlphaCentroidInit"){
    stream >> svmSettings.alphaCentroidInit;   
  }else{
    cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
    exit(-1);
  }
  
  stream >> setting;
  if(setting == "nIterations"){
    stream >> svmSettings.nIterations;   
  }else{
    cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
    exit(-1);
  }
  
  stream >> setting;
  if(setting == "learningRate"){
    stream >> svmSettings.learningRate;   
  }else{
    cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
    exit(-1);
  }
  
  stream >> setting;
  if(setting == "SVM_C_Data"){
    stream >> svmSettings.SVM_C_Data;   
  }else{
    cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
    exit(-1);
  }
  
  stream >> setting;
  if(setting == "SVM_C_Centroid"){
    stream >> svmSettings.SVM_C_Centroid;   
  }else{
    cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
    exit(-1);
  }
  
  stream >> setting;
  if(setting == "Cost"){
    stream >> svmSettings.cost;   
  }else{
    cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
    exit(-1);
  }
  
  stream >> setting;
  if(setting == "D2"){
    stream >> svmSettings.D2;   
  }else{
    cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
    exit(-1);
  }
  
  stream >> setting;
  if(setting == "sigmaClassicSimilarity"){
    stream >> svmSettings.sigmaClassicSimilarity;   
  }else{
    cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
    exit(-1);
  }
  

}

void CSVMSettings::parseGeneralSettings(ifstream& stream){
   string type, value;
   
   stream >> type;
   if(type != "type"){
      cout << "csvm::CSVMSettings.readGeneralSettings: Error! invalid settingsfile layout. Exitting..\n";
      exit(0);
   }
   stream >> value;
   
   if(value == "SVM")
      classifier = CL_SVM;
   else if(value == "CSVM")
      classifier == CL_CSVM;
   else if(value == "LINNET")
      classifier == CL_LINNET;
   else{
      cout << "csvm::parseGeneralSettings: " << value << " is not a recognized classifier method. Exitting..\n";
      exit(0);
   }
}

void CSVMSettings::readSettingsFile(string dir){
   ifstream file(dir.c_str(),ios::in);
   string line;

   
   if(!file.is_open()){
      cout << "csvm::CSVMSettings.readSettingsFile(" << dir << ") Error! Could not open settingsfile..\n";
      exit(0);
   }
   
   while(getline(file,line) && line != "Dataset");
   parseDatasetSettings(file);
   /*while(getline(file,line) && line != "ClusterAnalyser");
   parseClusterAnalserData(file);*/
   while(getline(file,line) && line != "Classifier");
   parseGeneralSettings(file);
   while(getline(file,line) && line != "Codebook");
   parseCodebookSettings(file);
   while(getline(file,line) && line != "FeatureExtractor");
   parseFeatureExtractorSettings(file);
   while(getline(file,line) && line != "ImageScanner");
   parseImageScannerSettings(file);
   while(getline(file,line) && line != "SVM");
   parseSVMSettings(file);
   while(getline(file,line) && line != "LinNet");
   parseLinNetSettings(file);
   while(getline(file,line) && line != "ConvSVM");
   parseConvSVMSettings(file);
   // parse values:
   
   /*string temp;
   file >> temp >> svmSettings.alpha;
   file >> temp >> svmSettings.beta;
   file >> temp >> svmSettings.COST;
   file >> temp >> svmSettings.D2;
   file >> temp >> svmSettings.SVM_C;
   file >> temp >> svmSettings.ALPHA_ITER;
   file >> temp >> svmSettings.NR_REP1;
   file >> temp >> svmSettings.NR_REP2;
   file >> temp >> svmSettings.EPS;
   file >> temp >> svmSettings.SIGMA;
   file >> temp >> svmSettings.INIT_ALPHA;*/
   
   
   file.close();   
}