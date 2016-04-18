#include <csvm/csvm_settings.h>

using namespace std;
using namespace csvm;

/* This settingsfile class parses the settingsfile, and wraps the info up to pass them 
 * to the other classes in the program that require them.
 * 
 * The methods of this class will be called by "CSVMClassifier in csvm_classifier.cc"
 * This is where the actual settings will be passed to the other classes.
 * 
 * The settings are put in a 'struct', so there each time one block of memory passed.
 * These structs are defined in the header files of the class where the're needed.
 * 
 * e.g. csvm_conv_svm.h contains the struct for convSVMSettings;
 * 
 * 
 */

CSVMSettings::~CSVMSettings() {
   //free(analyserSettings.rbmSettings.layerSizes);


}


//Some often used low level parsing private methods:
void CSVMSettings::parseUInt(unsigned int& intVal, ifstream& stream, string setting, string error){
  string s;
  stream >> s;
  if(s != setting)
    cout << error;
  stream >> intVal;
}

void CSVMSettings::parseDouble(double& doubleVal, ifstream& stream, string setting, string error){
  string s;
  stream >> s;
  if(s != setting)
    cout << error;
  stream >> doubleVal;
}

void CSVMSettings::parseBool(bool& boolVal, ifstream& stream, string setting, string error){
  string s;
  stream >> s;
  if(s != setting)
    cout << error;
  stream >> s;
  boolVal = (s == "TRUE" || s == "True" || s == "true" || s == "T" || s == "t" || s == "1" || s == "Y" || s == "y");
}

vector<string> CSVMSettings::parseStringArray(ifstream& stream, string setting, string error){
  vector<string> elements;
  vector<unsigned int> values;
  string s;
  char c;
  bool isEnded = false;
  
  stream >> s;
  if(s != setting){
    cout << error;
    exit(-1);
  }
  for(stream >> c; c != '['; stream >> c){
    if(c == '\n'){
      cout << error;
      exit(-1);
    }
  }
  for(size_t elementIdx = 0; !isEnded; ++elementIdx){
    string element = "";
    for(stream >> c; true; stream >> c){
      //skip spaces
      if(c == ' ')
        continue;
      //stop when ']' is read
      if(c == ']'){
        elements.push_back(element);
        isEnded = true;
        break;
      }
      if(c == ','){
        elements.push_back(element);
        break;
      }
      element += c;
    }
  }
  return elements;
}

vector<unsigned int> CSVMSettings::parseUIntArray(ifstream& stream, string setting, string error){
  vector<unsigned int> values;
  vector<string> elements = parseStringArray(stream, setting, error);
  unsigned int nElements = elements.size();
  unsigned int buf;
  
  for(size_t idx = 0; idx != nElements; ++idx){
    buf = stoi(elements[idx]);
    values.push_back(buf);
  }
  return values;
}

vector<int> CSVMSettings::parseIntArray(ifstream& stream, string setting, string error){
  vector<int> values;
  vector<string> elements = parseStringArray(stream, setting, error);
  unsigned int nElements = elements.size();
  int buf;
  
  for(size_t idx = 0; idx != nElements; ++idx){
    buf = stoi(elements[idx]);
    values.push_back(buf);
  }
  return values;
}
vector<bool> CSVMSettings::parseBoolArray(ifstream& stream, string setting, string error){
  vector<bool> values;
  vector<string> elements = parseStringArray(stream, setting, error);
  unsigned int nElements = elements.size();
  bool buf;
  
  for(size_t idx = 0; idx != nElements; ++idx){
    buf = (elements[idx] == "TRUE" || elements[idx] == "True" || elements[idx] == "true" || elements[idx] == "T" || elements[idx] == "t" || elements[idx] == "1" || elements[idx] == "Y" || elements[idx] == "y");  
    values.push_back(buf);
  }
  return values;
}

vector<double> CSVMSettings::parseDoubleArray(ifstream& stream, string setting, string error){
  vector<double> values;
  vector<string> elements = parseStringArray(stream, setting, error);
  unsigned int nElements = elements.size();
  double buf;
  
  for(size_t idx = 0; idx != nElements; ++idx){
    buf = stod(elements[idx]);
    values.push_back(buf);
  }
  return values;
}


//High level parsing methods

void CSVMSettings::parseConvSVMSettings(ifstream& stream) {
   string setting;
   string method;
   string value;
   string error = "csvm::csvm_settings:parseConvSVMSettings(): Error! Invalid settingsfile layout. Exitting...\n";
   
   parseBool(convSVMSettings.loadLastUsed, stream, "loadLastUsed", error);
   parseDouble(convSVMSettings.learningRate, stream, "learningRate", error);
   parseUInt(convSVMSettings.nIter, stream, "nIterations", error);
   parseDouble(convSVMSettings.initWeight, stream, "initWeight", error);
   parseDouble(convSVMSettings.CSVM_C, stream, "CSVM_C", error);
   parseBool(convSVMSettings.L2, stream, "L2", error);
}

void CSVMSettings::parseLinNetSettings(ifstream& stream) {
   string error = "csvm::csvm_settings:parseLinNetSettings(): Error! Invalid settingsfile layout. Exitting...\n";
   
   parseUInt(netSettings.nIter, stream, "nIterations", error);
   parseDouble(netSettings.initWeight, stream, "initWeight", error);
   parseDouble(netSettings.learningRate, stream, "learningRate", error);

}

void CSVMSettings::parseDatasetSettings(ifstream& stream) {

   string error = "csvm::csvm_settings:parseDatasetSettings(): Error! Invalid settingsfile layout. Exitting...\n";
   string setting;
   string method;
   
   stream >> setting;
   if (setting != "method") {
      cout << error;
      exit(-1);
   }
   stream >> method;
   if (method == "CIFAR10")
      datasetSettings.type = DATASET_CIFAR10;
   else if (method == "MNIST") 
      datasetSettings.type = DATASET_MNIST;
   
   parseUInt(datasetSettings.nTrainImages, stream, "nTrainImages", error);
   parseUInt(datasetSettings.nTestImages, stream, "nTestImages", error);
   parseUInt(datasetSettings.imWidth, stream, "imageWidth", error);
   parseUInt(datasetSettings.imHeight, stream, "imageHeight", error);

}


void CSVMSettings::parseCodebookSettings(ifstream& stream) {
   string error = "csvm::csvm_settings:parseDatasetSettings(): Error! Invalid settingsfile layout. Exitting...\n";
   string setting, method;
   
   parseBool(codebookSettings.generate, stream, "generate", error);
   parseBool(codebookSettings.standardize, stream, "standardize", error);
   parseBool(codebookSettings.whitening, stream, "whitening", error);
   parseUInt(codebookSettings.rootNPartitions, stream, "rootNPartitions", error);
 
   
   stream >> setting;
   if (setting != "method") {
      cout << "csvm::csvm_settings:parseCodebookSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }
   stream >> method;
   if (method == "LVQ") {
      codebookSettings.method = LVQ_Clustering;
   } else if (method == "KMEANS") {
      codebookSettings.method = KMeans_Clustering;
   } else if (method == "AKMEANS") {
      codebookSettings.method = AKMeans_Clustering;
   }
   
   parseUInt(codebookSettings.numberVisualWords, stream, "nClusters", error);
   parseUInt(codebookSettings.kmeansSettings.nIter, stream, "nIterations", error);
   dcbSettings.nIter = codebookSettings.kmeansSettings.nIter;
   
   //parse SimilarityFunction enum
   stream >> setting;
   if (setting == "SimilarityFunction") {
      stream >> method;
      if (method == "RBF") {
         codebookSettings.simFunction = CB_RBF;
         dcbSettings.simFunction = DCB_RBF;
      }
      else if (method == "SOFT_ASSIGNMENT") {
         codebookSettings.simFunction = SOFT_ASSIGNMENT;
         dcbSettings.simFunction = DCB_SOFT_ASSIGNMENT;
      }
      else if(method == "COSINE_SOFT_ASSIGNMENT"){
         codebookSettings.simFunction = COSINE_SOFT_ASSIGNMENT;
         dcbSettings.simFunction = DCB_COSINE_SOFT_ASSIGNMENT;
      }
      else
         cout << "Invalid codebook SimilarityFunction: Try RBF or SOFT_ASSIGNMENT\n";
   }
   else {
      cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }

   parseDouble(codebookSettings.similaritySigma, stream, "similaritySigma", error);
   dcbSettings.similaritySigma = codebookSettings.similaritySigma;

}


void CSVMSettings::parseImageScannerSettings(ifstream& stream) {
      string error = "csvm::csvm_settings:parseImageScannerSettings(): Error! Invalid settingsfile layout... Exitting...\n";
      
      parseUInt(scannerSettings.patchHeight, stream, "patchSize", error);
      scannerSettings.patchWidth = scannerSettings.patchHeight;
      parseUInt(scannerSettings.stride, stream, "scanStride", error);
      parseUInt(scannerSettings.nRandomPatches, stream, "nRandomPatches", error);
      dcbSettings.nRandomPatches = scannerSettings.nRandomPatches;
}



void CSVMSettings::parseFeatureExtractorSettings(ifstream& stream) {
	string setting;
	string method;
	string enumeration;
	string useColour;
  string error = "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
	
	nHOG = 0;
	nLBP = 0;
	nClean = 0;
  
  vector<string> tokens = parseStringArray(stream, "method", error);
  unsigned int nTokens = tokens.size();
  for(size_t tIdx = 0; tIdx != nTokens; ++tIdx){
    if(tokens[tIdx] == "CLEAN" || tokens[tIdx] == "clean" || tokens[tIdx] == "raw" || tokens[tIdx] == "RAW"){
      featureSettings.featureType.push_back(CLEAN);
      ++nClean;
    }
    else if(tokens[tIdx] == "HOG" || tokens[tIdx] == "hog"){
      featureSettings.featureType.push_back(HOG);
      ++nHOG;
    }
    else if(tokens[tIdx] == "LBP" || tokens[tIdx] == "lbp"){
      featureSettings.featureType.push_back(LBP);
      ++nLBP;
    }
  }
  nCodebooks = nClean + nHOG + nLBP;
  featureSettings.hogSettings.resize(nHOG);
  featureSettings.lbpSettings.resize(nLBP);
  featureSettings.clSettings.resize(nClean);

}



void CSVMSettings::parseHogSettings(ifstream& stream) {

    string error = "csvm::csvm_settings:parseHOGSettings(): Error reading settings... Exitting...\n";
    
    vector<unsigned int> uintArBuf;
    vector<string> strArBuf;
    vector<bool> boolArBuf;
    vector<int> intArBuf;
    
    for(size_t hIdx = 0; hIdx != nHOG; ++hIdx)
      featureSettings.hogSettings[hIdx].patchSize = scannerSettings.patchHeight;
    
    uintArBuf = parseUIntArray(stream, "nBins", error);
		for(size_t idx = 0; idx != nHOG && idx != uintArBuf.size(); ++idx){
      featureSettings.hogSettings[idx].nBins = uintArBuf[idx];
    }

    uintArBuf = parseUIntArray(stream, "cellSize", error);
    for(size_t idx = 0; idx != nHOG && idx != uintArBuf.size(); ++idx){
      featureSettings.hogSettings[idx].cellSize = uintArBuf[idx];
    }
    
    uintArBuf = parseUIntArray(stream, "cellStride", error);
    for(size_t idx = 0; idx != nHOG && idx != uintArBuf.size(); ++idx){
      featureSettings.hogSettings[idx].cellStride = uintArBuf[idx];
    }
		
    strArBuf = parseStringArray(stream, "padding", error);
    for(size_t idx = 0; idx != nHOG && idx != strArBuf.size(); ++idx){
      if(strArBuf[idx] == "None" || strArBuf[idx] == "none" || strArBuf[idx] == "NONE")
        featureSettings.hogSettings[idx].padding = NONE;
      else if(strArBuf[idx] == "Identity" || strArBuf[idx] == "identity" || strArBuf[idx] == "IDENTITY")
        featureSettings.hogSettings[idx].padding = IDENTITY;
      else if(strArBuf[idx] == "Zero" || strArBuf[idx] == "zero" || strArBuf[idx] == "ZERO")
        featureSettings.hogSettings[idx].padding = ZERO;
    }
		
    boolArBuf = parseBoolArray(stream, "useColourPixel", error);
    for(size_t idx = 0; idx != nHOG && idx != boolArBuf.size(); ++idx){
      featureSettings.hogSettings[idx].useColourPixel = boolArBuf[idx];
    }
    
    strArBuf = parseStringArray(stream, "interpolation", error);
    for(size_t idx = 0; idx != nHOG && idx != strArBuf.size(); ++idx){
      if(strArBuf[idx] == "INTERPOLATE_BINARY" || strArBuf[idx] == "binary" || strArBuf[idx] == "BINARY" || strArBuf[idx] == "Binary")
        featureSettings.hogSettings[idx].interpol = INTERPOLATE_BINARY;
      else if(strArBuf[idx] == "INTERPOLATE_LINEAR" || strArBuf[idx] == "linear" || strArBuf[idx] == "LINEAR" || strArBuf[idx] == "Linear")
        featureSettings.hogSettings[idx].interpol = INTERPOLATE_LINEAR;
    }
    
    strArBuf = parseStringArray(stream, "binmethod", error);
    for(size_t idx = 0; idx != nHOG && idx != strArBuf.size(); ++idx){
      if(strArBuf[idx] == "CROSSCOLOUR" || strArBuf[idx] == "CrossColour" || strArBuf[idx] == "crosscolour" || strArBuf[idx] == "Crosscolour")
        featureSettings.hogSettings[idx].binmethod = CROSSCOLOUR;
      else if(strArBuf[idx] == "BYCOLOUR" || strArBuf[idx] == "ByColour" || strArBuf[idx] == "bycolour" || strArBuf[idx] == "Bycolour")
        featureSettings.hogSettings[idx].binmethod = BYCOLOUR;
    }
    
    strArBuf = parseStringArray(stream, "postprocessing", error);
    for(size_t idx = 0; idx != nHOG && idx != strArBuf.size(); ++idx){
      if(strArBuf[idx] == "PURE" || strArBuf[idx] == "Pure" || strArBuf[idx] == "pure")
        featureSettings.hogSettings[idx].postproccess = PURE;
      else if(strArBuf[idx] == "STANDARDISATION" || strArBuf[idx] == "Standardise" || strArBuf[idx] == "std" || strArBuf[idx] == "STD")
        featureSettings.hogSettings[idx].postproccess = STANDARDISATION;
      else if(strArBuf[idx] == "NORMALISATION" || strArBuf[idx] == "norm" || strArBuf[idx] == "Norm" || strArBuf[idx] == "NORM")
        featureSettings.hogSettings[idx].postproccess = NORMALISATION;
      else if(strArBuf[idx] == "LTWONORM" || strArBuf[idx] == "L2NORM" || strArBuf[idx] == "L2" || strArBuf[idx] == "l2norm")
        featureSettings.hogSettings[idx].postproccess = LTWONORM;
      else if(strArBuf[idx] == "CLIPNORM" || strArBuf[idx] == "clipping" || strArBuf[idx] == "clipnorm" || strArBuf[idx] == "cnorm")
        featureSettings.hogSettings[idx].postproccess = CLIPNORM;
    }
  
    intArBuf = parseIntArray(stream, "debugLevel", error);
    for(size_t idx = 0; idx != nHOG && idx != intArBuf.size(); ++idx){
      featureSettings.hogSettings[idx].debugLevel = intArBuf[idx];
    }

	
}





void CSVMSettings::parseLBPSettings(ifstream& stream) {
    string error = "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
    vector<unsigned int> uintArBuf;
    vector<string> strArBuf;
    vector<bool> boolArBuf;
    
    for (size_t idx = 0; idx < nLBP; ++idx) {
      featureSettings.lbpSettings[idx].patchSize = scannerSettings.patchHeight;
    }
    
    uintArBuf = parseUIntArray(stream, "cellSize", error);
    for(size_t idx = 0; idx != uintArBuf.size() && idx != nLBP; ++idx){
      featureSettings.lbpSettings[idx].cellSize = uintArBuf[idx];
    }
    
    uintArBuf = parseUIntArray(stream, "cellStride", error);
    for(size_t idx = 0; idx != uintArBuf.size() && idx != nLBP; ++idx){
      featureSettings.lbpSettings[idx].cellStride = uintArBuf[idx];
    }
    
    strArBuf = parseStringArray(stream, "padding", error);
    for(size_t idx = 0; idx != nLBP && idx != strArBuf.size(); ++idx){
      if(strArBuf[idx] == "None" || strArBuf[idx] == "none" || strArBuf[idx] == "NONE")
        featureSettings.lbpSettings[idx].padding = LNONE;
      else if(strArBuf[idx] == "Identity" || strArBuf[idx] == "identity" || strArBuf[idx] == "IDENTITY")
        featureSettings.lbpSettings[idx].padding = LIDENTITY;
      else if(strArBuf[idx] == "Zero" || strArBuf[idx] == "zero" || strArBuf[idx] == "ZERO")
        featureSettings.lbpSettings[idx].padding = LZERO;
    }

    boolArBuf = parseBoolArray(stream, "useColourPixel", error);
    for(size_t idx = 0; idx != nHOG && idx != boolArBuf.size(); ++idx){
      featureSettings.lbpSettings[idx].useColourPixel = boolArBuf[idx];
    }

    strArBuf = parseStringArray(stream, "useUniformity", error);
    for(size_t idx = 0; idx != nLBP && idx != strArBuf.size(); ++idx){
      if(strArBuf[idx] == "UNIFORM" || strArBuf[idx] == "True" || strArBuf[idx] == "true")
        featureSettings.lbpSettings[idx].uniform = LUNIFORM;
      else if(strArBuf[idx] == "false" || strArBuf[idx] == "False" || strArBuf[idx] == "PURE")
        featureSettings.lbpSettings[idx].uniform = LPURE;
    }
    
    strArBuf = parseStringArray(stream, "binmethod", error);
    for(size_t idx = 0; idx != nLBP && idx != strArBuf.size(); ++idx){
      if(strArBuf[idx] == "CROSSCOLOUR" || strArBuf[idx] == "CrossColour" || strArBuf[idx] == "crosscolour" || strArBuf[idx] == "Crosscolour")
        featureSettings.lbpSettings[idx].binmethod = LCROSSCOLOUR;
      else if(strArBuf[idx] == "BYCOLOUR" || strArBuf[idx] == "ByColour" || strArBuf[idx] == "bycolour" || strArBuf[idx] == "Bycolour")
        featureSettings.lbpSettings[idx].binmethod = LBYCOLOUR;
    }


}

void CSVMSettings::parseCleanSettings(ifstream& stream) {
  string error = "csvm::CSVMSettings.parseCleanDescrSettings: Error! Invalid settingsfile layout. Exitting..\n";
  vector<string> strArBuf;

  strArBuf = parseStringArray(stream, "standardize", error);
  for(size_t idx = 0; idx != nClean && idx != strArBuf.size(); ++idx){
    if(strArBuf[idx] == "None")
      featureSettings.clSettings[idx].stdOptions = CL_NONE;
    else if(strArBuf[idx] == "PER_CHANNEL")
      featureSettings.clSettings[idx].stdOptions = CL_PER_CHANNEL;
    else if(strArBuf[idx] == "ALL")
      featureSettings.clSettings[idx].stdOptions = CL_ALL;
    else
      featureSettings.clSettings[idx].stdOptions = CL_NONE;
  }
}

void CSVMSettings::parseSVMSettings(ifstream& stream) {
   string setting;
   string method;
   string error = "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout... Exitting...\n";

   stream >> setting;
   if (setting == "Kernel") {
      stream >> method;
      if (method == "RBF")
         svmSettings.kernelType = RBF;
      else if (method == "LINEAR")
         svmSettings.kernelType = LINEAR;
      else
         cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";

   }
   else {
      cout << "csvm::csvm_settings:parseSVMSettings(): Error! Invalid settingsfile layout. Reading " << setting << ".. Exitting...\n";
      exit(-1);
   }

   parseDouble(svmSettings.alphaDataInit, stream, "AlphaDataInit", error);
   parseUInt(svmSettings.nIterations, stream, "nIterations", error);
   parseDouble(svmSettings.learningRate, stream, "learningRate", error);
   parseDouble(svmSettings.SVM_C_Data, stream, "SVM_C_Data", error);
   parseDouble(svmSettings.cost, stream, "Cost", error);
   parseDouble(svmSettings.D2, stream, "D2", error);
   parseDouble(svmSettings.sigmaClassicSimilarity, stream, "sigmaClassicSimilarity", error);

}

void CSVMSettings::parseGeneralSettings(ifstream& stream) {
   string type, value;
   string error = "csvm::CSVMSettings.readGeneralSettings: Error! invalid settingsfile layout. Exitting..\n";
   
   stream >> type;
   if (type != "Classifier") {
      cout << error;
      exit(0);
   }
   stream >> value;
   if (value == "SVM")
      classifier = CL_SVM;
   else if (value == "CSVM")
      classifier = CL_CSVM;
   else if (value == "LINNET")
      classifier = CL_LINNET;
   else {
      cout << "csvm::parseGeneralSettings: " << value << " is not a recognized classifier method. Exitting..\n";
      exit(0);
   }

   stream >> type;
   if (type != "Codebook") {
      cout << "csvm::CSVMSettings.readGeneralSettings: Error! invalid settingsfile layout. Exitting..\n";
      exit(0);
   }
   stream >> value;
   if (value == "CODEBOOK") {
      codebook = CB_CODEBOOK;
   }
   else if (value == "DEEPCODEBOOK") {
      codebook = CB_DEEPCODEBOOK;
   }
   else {
      cout << "csvm::parseGeneralSettings: " << value << " is not a recognized codebook method. Exitting..\n";
      exit(0);
   }
   
   parseUInt(netSettings.nClasses, stream, "nClasses", error);
   convSVMSettings.nClasses = netSettings.nClasses;
   datasetSettings.nClasses = netSettings.nClasses;
   parseBool(debugOut, stream, "debugOut", error);
   parseBool(normalOut, stream, "normalOut", error);
   parseBool(liveROut, stream, "liveROut", error);
   codebookSettings.kmeansSettings.liveROut = liveROut;
}

/*void CSVMSettings::parseCleanDescrSettings(ifstream& stream){
   string type, value;
   stream >> type;
   if(type != "standardize"){
      cout << "csvm::CSVMSettings.parseCleanDescrSettings: Error! Invalid settingsfile layout. Exitting..\n";
      exit(0);
   }
   stream >> value;
   if(value == "None")
      featureSettings.clSettings.stdOptions = CL_NONE;
   else if(value == "PER_CHANNEL")
      featureSettings.clSettings.stdOptions = CL_PER_CHANNEL;
   else if(value == "ALL")
      featureSettings.clSettings.stdOptions = CL_ALL;
   else
      featureSettings.clSettings.stdOptions = CL_NONE;
}*/

void CSVMSettings::readSettingsFile(string dir) {
   ifstream file(dir.c_str(), ios::in);
   string line;


   if (!file.is_open()) {
      cout << "csvm::CSVMSettings.readSettingsFile(" << dir << ") Error! Could not open settingsfile..\n";
      exit(0);
   }

   while (getline(file, line) && line != "Dataset");
   parseDatasetSettings(file);
   /*while(getline(file,line) && line != "ClusterAnalyser");
   parseClusterAnalserData(file);*/
   while (getline(file, line) && line != "General");
   parseGeneralSettings(file);
   while (getline(file, line) && line != "Codebook");
   parseCodebookSettings(file);
   while (getline(file, line) && line != "ImageScanner");
   parseImageScannerSettings(file);
   while (getline(file, line) && line != "FeatureExtractor");
   parseFeatureExtractorSettings(file);
   //while(getline(file, line) && line != "CleanDescriptor");
   //parseCleanDescrSettings(file);
   if(nClean > 0){
      while (getline(file, line) && line != "CLEAN");
      parseCleanSettings(file);
   }
   
   if(nHOG > 0){
      while (getline(file, line) && line != "HOG");
      parseHogSettings(file);
   }
   if(nLBP > 0){
      while (getline(file, line) && line != "LBP");
      parseLBPSettings(file);
   }
   
   while (getline(file, line) && line != "SVM");
   parseSVMSettings(file);
   while (getline(file, line) && line != "LinNet");
   parseLinNetSettings(file);
   while (getline(file, line) && line != "ConvSVM");
   parseConvSVMSettings(file);
   // parse values:

   file.close();
}

