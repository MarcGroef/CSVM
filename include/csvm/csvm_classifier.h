#ifndef CSVM_CLASSIFIER_H
#define CSVM_CLASSIFIER_H

/* This class implements the general pipeline of the system.
 * Based on the settingsfile, it will construct codebooks, describe images and train/test the system.
 * 
 * 
 * 
 * 
 */

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
      DeepCodebook* deepCodebook;
      ImageScanner imageScanner;
      //ClusterAnalyser analyser;
      FeatureExtractor featExtr;
      vector< SVM > svms;
      LinNetwork linNetwork;
      ConvSVM convSVM;
      
      vector< vector<double> > classicTrainActivations;
      
      vector< vector<double> > datasetActivations, validationActivations;
      
      vector<Feature> collectFeaturesFromImage(Image* im);
      vector<double> getActivationsFromImage(Image* im);
      
      void trainLinearNetwork();
      unsigned int lnClassify(Image* image);

      
      void trainConvSVMs();
      unsigned int classifyConvSVM(Image* im);
      
      void trainClassicSVMs();
      unsigned int classifyClassicSVMs(Image* im, bool printResults);
      
   public:
      //public vars
      CSVMDataset dataset;
      bool debugOut, normalOut;
      CSVMClassifier();
      ~CSVMClassifier();
      void setSettings(string settingsFile);
      void constructCodebook();
      void constructDeepCodebook();
      bool getGenerateCB();
      
      void exportCodebook(string filename);
      void importCodebook(string filename);
      
      void initSVMs();
      unsigned int getNoClasses();
      bool useOutput();

      void train();
      unsigned int classify(Image* im);
      unsigned int classifyFromActivation(unsigned int testIdx);
   };
   
}
#endif
