 #include <csvm/csvm_settings.h>

using namespace std;
using namespace csvm;

CSVMSettings::~CSVMSettings(){
  //free(analyserSettings.rbmSettings.layerSizes);
  
  
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


void CSVMSettings::parseClusterAnalserData(ifstream& stream){
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
    if(setting == "similaritySigma"){
      stream >> codebookSettings.similaritySigma;
    }else{
      cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
    }
    
    
    
  }
}

void CSVMSettings::parseFeatureExtractorSettings(ifstream& stream){
  string setting;
  string method;
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
  if(setting == "sigmaClassicSimilarity"){
    stream >> svmSettings.sigmaClassicSimilarity;   
  }else{
    cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
    exit(-1);
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
   while(getline(file,line) && line != "ClusterAnalyser");
   parseClusterAnalserData(file);
   while(getline(file,line) && line != "Codebook");
   parseCodebookSettings(file);
   while(getline(file,line) && line != "FeatureExtractor");
   parseFeatureExtractorSettings(file);
   while(getline(file,line) && line != "ImageScanner");
   parseImageScannerSettings(file);
   while(getline(file,line) && line != "SVM");
   parseSVMSettings(file);
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