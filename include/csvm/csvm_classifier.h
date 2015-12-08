#ifndef CSVM_CLASSIFIER_H
#define CSVM_CLASSIFIER_H

#include <vector>
#include <cstdlib>
#include <ctime>

#include "csvm_settings.h"
#include "csvm_dataset.h"
#include "csvm_codebook.h"
#include "csvm_image_scanner.h"
//#include "csvm_cluster_analyser.h"
#include "csvm_feature.h"
#include "csvm_feature_extractor.h"
#include "csvm_svm.h"
#include "csvm_linear_network.h"
#include "csvm_deep_codebook.h"
#include "csvm_conv_svm.h"

using namespace std;
namespace csvm{
   
   class CSVMClassifier{
      CSVMSettings settings;
      Codebook codebook;
      DeepCodebook deepCodebook;
      ImageScanner imageScanner;
      //ClusterAnalyser analyser;
      FeatureExtractor featExtr;
      vector< SVM > svms;
      LinNetwork linNetwork;
      bool standardizeActivations;
      ConvSVM convSVM;
      vector < vector< vector<double> > > classicTrainActivations;
   public:
      //public vars
      bool useLinNet;
      CSVMDataset dataset;
      
      CSVMClassifier();
      void setSettings(string settingsFile);
      void constructCodebook();
      void exportCodebook(string filename);
      void importCodebook(string filename);
      void trainSVMs();
      void trainClassicSVMs();
      void initSVMs();
      //unsigned int classify(Image* image);
      unsigned int getNoClasses();
      //CSVMClassifier();
      unsigned int classifyClassicSVMs(Image* im, bool printResults);
      bool useClassicSVM();
      void trainLinearNetwork();
      unsigned int lnClassify(Image* image);
      void constructDeepCodebook();
      void trainConvSVMs();
      unsigned int classifyConvSVM(Image* im);
      void train();
      unsigned int classify(Image* im);
   };
   
}
#endif
