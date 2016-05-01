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
  if(s != setting){
    cout << error;
    exit(-1);
  }
  stream >> intVal;
}

void CSVMSettings::parseDouble(double& doubleVal, ifstream& stream, string setting, string error){
  string s;
  stream >> s;
  if(s != setting){
    cout << error;
    exit(-1);
  }
  stream >> doubleVal;
}

void CSVMSettings::parseBool(bool& boolVal, ifstream& stream, string setting, string error){
  string s;
  stream >> s;
  if(s != setting){
    cout << error;
    exit(-1);
  }
  stream >> s;
  boolVal = (s == "TRUE" || s == "True" || s == "true" || s == "T" || s == "t" || s == "1" || s == "Y" || s == "y");
}

//high level parsing:

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
   dcbSettings.nCentroids = codebookSettings.numberVisualWords;
   parseUInt(codebookSettings.kmeansSettings.nIter, stream, "nIterations", error);
   dcbSettings.nIter = codebookSettings.kmeansSettings.nIter;
   
   //parse deep architecture enum
   stream >> setting;
   if (setting == "DeepArchitecture") {
      stream >> method;
      if(method == "ALPHA")
         dcbSettings.architecture = DCB_ALPHA;
      else if(method == "BETA")
         dcbSettings.architecture = DCB_BETA;
      else if(method == "GAMMA")
         dcbSettings.architecture = DCB_GAMMA;
   }else {
      cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }
   
   stream >> setting;
   if (setting == "DeepPoolingMethod") {
      stream >> method;
      if(method == "SUM")
         dcbSettings.poolmethod = DCB_MAX;
      else if(method == "MAX")
         dcbSettings.poolmethod = DCB_SUM;
   }else {
      cout << "csvm::csvm_settings:parseCodebookData(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }
   
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

void CSVMSettings::parseFeatureExtractorSettings(ifstream& stream) {
   string setting;
   string method;
   string enumeration;
   string useColour;
   stream >> setting;
   if (setting != "method") {
      cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
      exit(-1);
   }
   stream >> method;
   if (method == "LBP") {
      featureSettings.featureType = LBP;


      stream >> setting;
      if (setting == "cellSize") {  // #cellSize is best an even-numbered, divisor of patch size. By default it'll be half of patch size
          stream >> featureSettings.lbpSettings.cellSize;

      }
      else {
          cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
          exit(-1);
      }

      stream >> setting;
      if (setting == "cellStride") { //#cellStride is best an even-numbered, divisor of cellSize. By default it's the same value as cellSize, meaning the patch is divided into quadrants, and not iterated over 
          stream >> featureSettings.lbpSettings.cellStride;

      }
      else {
          cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
          exit(-1);
      }

      stream >> setting;
      if (setting == "patchSize") {  // 
          stream >> featureSettings.lbpSettings.patchSize;
      }
      else {
          cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! HOG patchSize not specified! Invalid settingsfile layout. Exitting...\n";
          exit(-1);
      }


      stream >> setting;
      if (setting == "padding") {//#the size of a patch
          stream >> enumeration;
          if (enumeration == "None" || enumeration == "none" || enumeration == "NONE")
              featureSettings.lbpSettings.padding = LNONE;
          else if (enumeration == "Identity" || enumeration == "identity" || enumeration == "IDENTITY")
              featureSettings.lbpSettings.padding = LIDENTITY;
          else if (enumeration == "Zero" || enumeration == "zero" || enumeration == "ZERO")
              featureSettings.lbpSettings.padding = LZERO;

      }
      else {
          cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
          exit(-1);
      }

      stream >> setting;
      if (setting == "useColourPixel") {//if we use grey images
          stream >> useColour;
          if (useColour == "true" || useColour == "True")
              featureSettings.lbpSettings.useColourPixel = true;
          else {
              if (useColour == "false" || useColour == "False")
                  featureSettings.lbpSettings.useColourPixel = false;
          }
      }
      else {
          cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
          exit(-1);
      }

      stream >> setting;
      if (setting == "useUniformity") {//if we use grey images
          stream >> enumeration;
          if (enumeration == "UNIFORM" || enumeration == "True" || enumeration == "true")
              featureSettings.lbpSettings.uniform = LUNIFORM;
          else {
              if (enumeration == "false" || enumeration == "false" || enumeration == "PURE")
                  featureSettings.lbpSettings.uniform = LPURE;
          }
      }
      else {
          cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
          exit(-1);
      }

      stream >> setting;
      if (setting == "binmethod") {  // things
          stream >> enumeration;
          if (enumeration == "CROSSCOLOUR" || enumeration == "CrossColour" || enumeration == "crosscolour" || enumeration == "Crosscolour")
              featureSettings.lbpSettings.binmethod = LCROSSCOLOUR;
          else {
              if (enumeration == "BYCOLOUR" || enumeration == "ByColour" || enumeration == "bycolour" || enumeration == "Bycolour")
                  featureSettings.lbpSettings.binmethod = LBYCOLOUR;
          }

      }
      else {
          cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error at binmethod! Invalid settingsfile layout. Exitting...\n";
          exit(-1);
      }



   }
   else if (method == "HOG") {
      featureSettings.featureType = HOG;

      stream >> setting;
      if (setting == "nBins") {  // #nbins is conventionally 9, but can be different.
         stream >> featureSettings.hogSettings.nBins;
      }
      else {
         cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! HOG nBins not set! Exitting...\n";
         exit(-1);
      }

      stream >> setting;
      if (setting == "cellSize") {  // #cellSize is best an even-numbered, divisor of patch size. By default it'll be half of patch size
         stream >> featureSettings.hogSettings.cellSize;

      }
      else {
         cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

      stream >> setting;
      if (setting == "cellStride") { //#cellStride is best an even-numbered, divisor of cellSize. By default it's the same value as cellSize, meaning the patch is divided into quadrants, and not iterated over 
         stream >> featureSettings.hogSettings.cellStride;

      }
      else {
         cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

      stream >> setting;
      if (setting == "patchSize") {  // 
         stream >> featureSettings.hogSettings.patchSize;
      }
      else {
         cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! HOG patchSize not specified! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

      stream >> setting;
      if (setting == "padding") {//#the size of a patch
         stream >> enumeration;
         if (enumeration == "None" || enumeration == "none" || enumeration == "NONE")
            featureSettings.hogSettings.padding = NONE;
         else if (enumeration == "Identity" || enumeration == "identity" || enumeration == "IDENTITY")
            featureSettings.hogSettings.padding = IDENTITY;
         else if (enumeration == "Zero" || enumeration == "zero" || enumeration == "ZERO")
            featureSettings.hogSettings.padding = ZERO;

      }
      else {
         cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

      stream >> setting;
      if (setting == "useColourPixel") {//if we use grey images
         stream >> useColour;
         if (useColour == "true" || useColour == "True")
            featureSettings.hogSettings.useColourPixel = true;
         else {
            if (useColour == "false" || useColour == "False")
               featureSettings.hogSettings.useColourPixel = false;
         }
      }
      else {
         cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

      stream >> setting;
      if (setting == "interpolation") {  // things
         stream >> enumeration;
         if (enumeration == "INTERPOLATE_BINARY" || enumeration == "binary" || enumeration == "BINARY" || enumeration == "Binary")
            featureSettings.hogSettings.interpol = INTERPOLATE_BINARY;
         else {
            if (enumeration == "INTERPOLATE_LINEAR" || enumeration == "linear" || enumeration == "LINEAR" || enumeration == "Linear")
               featureSettings.hogSettings.interpol = INTERPOLATE_LINEAR;
         }

      }
      else {
         cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error at interpolationmethod! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

      stream >> setting;
      if (setting == "binmethod") {  // things
         stream >> enumeration;
         if (enumeration == "CROSSCOLOUR" || enumeration == "CrossColour" || enumeration == "crosscolour" || enumeration == "Crosscolour")
            featureSettings.hogSettings.binmethod = CROSSCOLOUR;
         else {
            if (enumeration == "BYCOLOUR" || enumeration == "ByColour" || enumeration == "bycolour" || enumeration == "Bycolour")
               featureSettings.hogSettings.binmethod = BYCOLOUR;
         }

      }
      else {
         cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error at binmethod! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

      stream >> setting;
      if (setting == "postprocessing") {  // things
         stream >> enumeration;
         if (enumeration == "PURE" || enumeration == "Pure" || enumeration == "pure")
            featureSettings.hogSettings.postproccess = PURE;
         if (enumeration == "STANDARDISATION" || enumeration == "Standardise" || enumeration == "standardise" || enumeration == "std" || enumeration == "STD")
               featureSettings.hogSettings.postproccess = STANDARDISATION;
         if (enumeration == "NORMALISATION" || enumeration == "norm" || enumeration == "Norm" || enumeration == "NORM")
            featureSettings.hogSettings.postproccess = NORMALISATION;
         if (enumeration == "LTWONORM" || enumeration == "L2NORM" || enumeration == "L2" || enumeration == "l2norm")
            featureSettings.hogSettings.postproccess = LTWONORM;
         if (enumeration == "CLIPNORM" || enumeration == "clipping" || enumeration == "clipnorm" || enumeration == "cnorm")
            featureSettings.hogSettings.postproccess = CLIPNORM;
      }
      else {
         cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! at HOG postprocessing settings Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

      stream >> setting;
      if (setting == "debugLevel") {  // things
         stream >> enumeration;
         if (enumeration == "-1" )
            featureSettings.hogSettings.debugLevel = -1;
         if (enumeration == "0")
            featureSettings.hogSettings.debugLevel = 0;
         if (enumeration == "1")
            featureSettings.hogSettings.debugLevel = 1;
         if (enumeration == "2")
            featureSettings.hogSettings.debugLevel = 2;
      }
      else {
         cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error at binmethod! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

   }
   else if (method == "CLEAN") {
      featureSettings.featureType = CLEAN;
   }
   else if (method == "PIXHOG") {
      featureSettings.featureType = MERGE;

      stream >> setting;
      if (setting == "cellSize") {  // #cellSize is best an even-numbered, divisor of patch size. By default it'll be half of patch size
         stream >> featureSettings.hogSettings.cellSize;
      }
      else {
         cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

      stream >> setting;
      if (setting == "cellStride") { //#cellStride is best an even-numbered, divisor of cellSize. By default it's the same value as cellSize, meaning the patch is divided into quadrants, and not iterated over 
         stream >> featureSettings.hogSettings.cellStride;
      }
      else {
         cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }


      stream >> setting;
      if (setting == "padding") {//#the size of a patch
         stream >> enumeration;
         if (enumeration == "None")
            featureSettings.hogSettings.padding = NONE;
         else if (enumeration == "Identity")
            featureSettings.hogSettings.padding = IDENTITY;
         else if (enumeration == "Zero")
            featureSettings.hogSettings.padding = ZERO;

      }
      else {
         cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }

      stream >> setting;
      if (setting == "useColourPixel") {//if we use grey images
         stream >> useColour;
         if (useColour == "true") {
            featureSettings.hogSettings.useColourPixel = true;
            featureSettings.mergeSettings.useColourPixel = true;
         }
         else {
            if (useColour == "false") {
               featureSettings.hogSettings.useColourPixel = false;
               featureSettings.mergeSettings.useColourPixel = false;
            }
         }
      }
      else {
         cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }
      stream >> setting;
      if (setting == "weightRatio") { //#cellStride is best an even-numbered, divisor of cellSize. By default it's the same value as cellSize, meaning the patch is divided into quadrants, and not iterated over 
         stream >> featureSettings.mergeSettings.weightRatio;
      }
      else {
         cout << "csvm::csvm_settings:parseFeatureExtractorSettings(): Error! Invalid settingsfile layout. Exitting...\n";
         exit(-1);
      }
   }



}

void CSVMSettings::parseImageScannerSettings(ifstream& stream) {
      string error = "csvm::csvm_settings:parseImageScannerSettings(): Error! Invalid settingsfile layout... Exitting...\n";
      
      parseUInt(scannerSettings.patchHeight, stream, "patchHeight", error);
      parseUInt(scannerSettings.patchWidth, stream, "patchWidth", error);
      parseUInt(scannerSettings.stride, stream, "scanStride", error);
      parseUInt(scannerSettings.nRandomPatches, stream, "nRandomPatches", error);
      dcbSettings.nRandomPatches = scannerSettings.nRandomPatches;
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

void CSVMSettings::parseCleanDescrSettings(ifstream& stream){
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
}

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
   while (getline(file, line) && line != "FeatureExtractor");
   parseFeatureExtractorSettings(file);
   while(getline(file, line) && line != "CleanDescriptor");
   parseCleanDescrSettings(file);
   while (getline(file, line) && line != "ImageScanner");
   parseImageScannerSettings(file);
   while (getline(file, line) && line != "SVM");
   parseSVMSettings(file);
   while (getline(file, line) && line != "LinNet");
   parseLinNetSettings(file);
   while (getline(file, line) && line != "ConvSVM");
   parseConvSVMSettings(file);
   // parse values:

   file.close();
}

