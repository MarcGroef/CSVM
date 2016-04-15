#include <csvm/csvm_classifier.h>
#include <iomanip>

/* This class implements the general pipeline of the system.
 * Based on the settingsfile, it will construct codebooks, describe images and train/test the system.
 * 
 * 
 * 
 * 
 */

using namespace std;
using namespace csvm;

//initialize random
CSVMClassifier::CSVMClassifier(){
   srand(time(NULL)); 
   deepCodebook = new DeepCodebook(&featExtr, &imageScanner, &dataset);
}

CSVMClassifier::~CSVMClassifier(){
   delete deepCodebook;
}

bool CSVMClassifier::getGenerateCB(){
   return codebook.getGenerate();
}

/**Some often used functionality:*/

vector<Feature> CSVMClassifier::collectFeaturesFromImage(Image* im){
  vector<Feature> features; 
  vector<Patch> patches = imageScanner.scanImage(im);
  size_t nPatches = patches.size();
  for(size_t pIdx = 0; pIdx != nPatches; ++pIdx){
    features.push_back(featExtr.extract(patches[pIdx]));
  }
  return features;
}

vector<double> CSVMClassifier::getActivationsFromImage(Image* im){
  return codebook.getActivations(collectFeaturesFromImage(im));
}




/////////////////////////////////////


//initialize the SVMs, by telling them the dataset size, amount of classes, centroids, and the respective label of the SVM
void CSVMClassifier::initSVMs(){
   
   unsigned int ensembleSize = dataset.getNumberClasses();
   
   svms.reserve(dataset.getNumberClasses());
   for(size_t svmIdx = 0; svmIdx < ensembleSize; ++svmIdx){
      svms.push_back(SVM(dataset.getTrainSize(), codebook.getNClasses(), codebook.getNCentroids(), svmIdx));
      svms[svmIdx].setSettings(settings.svmSettings);
      svms[svmIdx].debugOut = settings.debugOut;
      svms[svmIdx].normalOut = settings.normalOut;
   }
   
}

//read settings file, and pass the settings to respective modules
void CSVMClassifier::setSettings(string settingsFile){
   settings.readSettingsFile(settingsFile);
   //analyser.setSettings(settings.analyserSettings);
   imageScanner.setSettings(settings.scannerSettings);
   imageScanner.debugOut = settings.debugOut;
   imageScanner.normalOut = settings.normalOut;
   
   dataset.setSettings(settings.datasetSettings);
   dataset.debugOut = settings.debugOut;
   
   dataset.normalOut = settings.normalOut;
   settings.codebookSettings.kmeansSettings.normalOut = settings.normalOut;
   codebook.setSettings(settings.codebookSettings);
   codebook.debugOut = settings.debugOut;
   codebook.normalOut = settings.normalOut;

   
   
   
   featExtr.setSettings(settings.featureSettings);
   featExtr.debugOut = settings.debugOut;
   featExtr.normalOut = settings.normalOut;
   
   settings.netSettings.nCentroids = settings.codebookSettings.numberVisualWords;
   
   linNetwork.setSettings(settings.netSettings);
   linNetwork.debugOut = settings.debugOut;
   linNetwork.normalOut = settings.normalOut;
   
   convSVM.setSettings(settings.convSVMSettings);
   convSVM.debugOut = settings.debugOut;
   convSVM.normalOut = settings.normalOut;
   
   
}

//Train the system
void CSVMClassifier::train(){
   if(settings.classifier == CL_SVM){
      if(settings.normalOut) cout << "Training SVM..\n";
         trainClassicSVMs();
      return;
   }
   
   unsigned int nTrainImages = dataset.getTrainSize();
   unsigned int nTestImages = dataset.getTestSize();
   vector < vector < double > > datasetActivations;
   vector < vector < double > > validationActivations;
   
   
   //get codebook activations for all train images
   for(size_t dataIdx = 0; dataIdx < nTrainImages; ++dataIdx){
      if(settings.codebook == CB_CODEBOOK){
        datasetActivations.push_back(getActivationsFromImage(dataset.getTrainImagePtr(dataIdx)));
     }else{
        datasetActivations.push_back(deepCodebook->getActivations(dataset.getTrainImagePtr(dataIdx)));
     }
   }
   cout << "activation size = " << datasetActivations[0].size() << endl;
   // this is enough for the linear network to train on, so if it is chosen, train it and return
   
   if(settings.classifier == CL_LINNET){
     linNetwork.train(datasetActivations, &dataset);
     return;
   }
   

   //get codebook activations for all test images, for validation stats
   for(size_t dataIdx = 0; dataIdx < nTestImages; ++dataIdx){
     if(settings.codebook == CB_CODEBOOK){
        validationActivations.push_back(getActivationsFromImage(dataset.getTestImagePtr(dataIdx)));
     }else{
        validationActivations.push_back(deepCodebook->getActivations(dataset.getTestImagePtr(dataIdx)));
     }
   }
   convSVM.settings.nCentroids = datasetActivations[0].size();
   
   if(!settings.convSVMSettings.loadLastUsed)
      convSVM.initialize();
   else
      convSVM.importFromFile("LAST_USED_CSVM");
  
   ofstream statFile;
   statFile.open( "INTERM_SCORE.csv" );
   system("Rscript ../genRplotsLive INTERM_SCORE &> ./logs/errorLOG");
   statFile << "TrainingRound,Score" << endl << "0,0" << endl;
 
   //All other classifiers are handled, so lets train the csvm, with validation stats
   size_t nTrainEpochs = settings.convSVMSettings.nIter;
   size_t nTrainsPerValidation = 50;
   cout << "trainSize = " << datasetActivations.size() << endl;
   for(size_t eIdx = 0; eIdx != nTrainEpochs;){
     if(eIdx + nTrainsPerValidation < nTrainEpochs){
       convSVM.train(datasetActivations, nTrainsPerValidation, &dataset);
       eIdx += nTrainsPerValidation;
     }
     else{
       convSVM.train(datasetActivations, nTrainEpochs - eIdx, &dataset);
       eIdx += nTrainEpochs - eIdx;
     }
     double validScore = convSVM.validate(validationActivations, &dataset);
     statFile << eIdx << "," << validScore << endl;
     if (settings.debugOut) cout << "epoch " << eIdx << "\t\tScore: " << validScore << endl;
   }
   statFile << "EOF" << endl;
   statFile.close();
   convSVM.exportToFile("LAST_USED_CSVM");
}

//Classify an image, given its pointer
unsigned int CSVMClassifier::classify(Image* im){
   unsigned int result = 0;
   
   switch(settings.classifier){
      case CL_SVM:
         result = classifyClassicSVMs(im, false); //return value should be processed
         break;
      case CL_CSVM:
         result = classifyConvSVM(im);
         break;
      case CL_LINNET:
         result = lnClassify(im);
         break;
   }
   return result;
}



//export the current codebook to file (Only works for the normal codebook, not yet for the deep bow)
void CSVMClassifier::exportCodebook(string filenamesstream){
   codebook.exportCodebook(filenamesstream);
}


//import the current codebook
void CSVMClassifier::importCodebook(string filename){
   codebook.importCodebook(filename);
}


void CSVMClassifier::constructDeepCodebook(){
   deepCodebook->setSettings(settings.dcbSettings);
   deepCodebook->debugOut = settings.debugOut;
   deepCodebook->normalOut = settings.normalOut;
   deepCodebook->generateCentroids();
   
   if(normalOut) cout << "Done constructing deep codebook\n";
}


//return number of classes
unsigned int CSVMClassifier::getNoClasses(){
   return dataset.getNumberClasses();
}


//construct a codebook using the current dataset
void CSVMClassifier::constructCodebook(){
   if(settings.codebook == CB_DEEPCODEBOOK){
      constructDeepCodebook();
      return;
   }
   //unsigned int nClasses = dataset.getNumberClasses();
   
   unsigned int nPatches = settings.scannerSettings.nRandomPatches;
   
   vector<Feature> pretrainDump;


   for(size_t pIdx = 0; pIdx < nPatches; ++pIdx){
      
      //patches = imageScanner.getRandomPatches(dataset.getImagePtrFromClass(im, cl));
      Patch patch = imageScanner.getRandomPatch(dataset.getImagePtr(rand() % dataset.getTotalImages()));
      Feature newFeat = featExtr.extract(patch);
      pretrainDump.push_back(newFeat);//insert(pretrainDump[cl].end(),features.begin(),features.end());

      
   }

   if(settings.debugOut) cout << "Collected " << pretrainDump.size()<< " features\n";
   codebook.constructCodebook(pretrainDump);
   codebook.exportToPNG(dataset.getTrainImagePtr(0)->getNChannels());
   if(settings.debugOut) cout << "done constructing codebook using "  << settings.scannerSettings.nRandomPatches << " patches\n";
   
   pretrainDump.clear();
}

//Not used anymore. All training is done in the train function itself.
void CSVMClassifier::trainConvSVMs(){
   unsigned int nTrainImages = dataset.getTrainSize();
   vector < vector < double > > datasetActivations;
   
   //allocate space for more vectors
   datasetActivations.reserve(nTrainImages);
   //for all trainings imagages:
   for(size_t dataIdx = 0; dataIdx < nTrainImages; ++dataIdx){
      if(settings.codebook == CB_CODEBOOK){
         datasetActivations.push_back(getActivationsFromImage(dataset.getTrainImagePtr(dataIdx)));
     }else{
        datasetActivations.push_back(deepCodebook->getActivations(dataset.getTrainImagePtr(dataIdx)));
     }
   }
   convSVM.train(datasetActivations, settings.convSVMSettings.nIter, &dataset);
}


unsigned int CSVMClassifier::classifyConvSVM(Image* image){
   vector<double> dataActivation;
   
   if(settings.codebook == CB_CODEBOOK){
      dataActivation = getActivationsFromImage(image);
   }else{      
      dataActivation = deepCodebook->getActivations(image);
   }

   return convSVM.classify(dataActivation);
}

bool CSVMClassifier::useOutput(){
   return settings.normalOut;

}

//train the regular-SVM
void CSVMClassifier::trainClassicSVMs(){
   unsigned int nTrainImages = dataset.getTrainSize();
   unsigned int nClasses = dataset.getNumberClasses(); 
   unsigned int nCentroids; 
   
   vector < vector<double> > datasetActivations;
   vector < Feature > dataFeatures;
   vector < Patch > patches;
   vector < vector<double> > dataKernel(nTrainImages);
   vector<double> dataActivation;

   //allocate space for more vectors
   datasetActivations.reserve(nTrainImages);
   //for all trainings imagages:
   for(size_t dataIdx = 0; dataIdx < nTrainImages; ++dataIdx){
      
      //extract patches
      if(settings.codebook == CB_CODEBOOK){
         datasetActivations.push_back(getActivationsFromImage(dataset.getTrainImagePtr(dataIdx)));

     }else{
        datasetActivations.push_back(deepCodebook->getActivations(dataset.getTrainImagePtr(dataIdx)));
     }
   }
   nCentroids = datasetActivations[0].size();
   
   if(settings.debugOut) cout << "nClasses = " << nClasses << endl;
   if(settings.debugOut) cout << "nCentroids = " << nCentroids << endl;
   
   if(settings.debugOut) cout << "Calculating similarities\n";
   //calculate similarity kernal between activation vectors
   for(size_t dIdx0 = 0; dIdx0 < nTrainImages; ++dIdx0){ 
      for(size_t dIdx1 = 0; dIdx1 <= dIdx0; ++dIdx1){
         dataKernel[dIdx0].resize(dIdx0 + 1);
         double sum = 0;
	 
         if(settings.svmSettings.kernelType == RBF){
            
	    sum = 0;
            for(size_t centr = 0; centr < nCentroids; ++centr){
               sum += (datasetActivations[dIdx0][centr] - datasetActivations[dIdx1][centr])*(datasetActivations[dIdx0][centr] - datasetActivations[dIdx1][centr]);
            }
            dataKernel[dIdx0][dIdx1] = exp((-1.0 * sum)/settings.svmSettings.sigmaClassicSimilarity);
            //dataKernel[dIdx1][dIdx0] = dataKernel[dIdx0][dIdx1];
            
         }else if (settings.svmSettings.kernelType == LINEAR){
 
	    sum = 0;
	    for(size_t centr = 0; centr < nCentroids; ++centr){
	      sum += (datasetActivations[dIdx0][centr] * datasetActivations[dIdx1][centr]);
	    }
            dataKernel[dIdx0][dIdx1] = sum;
         }else
            cout << "CSVM::svm::Error! No valid kernel type selected! Try: RBF or LINEAR\n"  ;
	 
      }
      cout << "done with similarity of " << dIdx0 << endl;
   }
   for(size_t cl = 0; cl < nClasses; ++cl){
      svms[cl].trainClassic(dataKernel, &dataset);  
   }
   classicTrainActivations = datasetActivations;
}


//classify an image using the regular-SVM
unsigned int CSVMClassifier::classifyClassicSVMs(Image* image, bool printResults){
   
   unsigned int nClasses = dataset.getNumberClasses();
   vector<double> dataActivation;
   
   if(settings.codebook == CB_CODEBOOK){
      dataActivation = getActivationsFromImage(image);
   }else{
      dataActivation = deepCodebook->getActivations(image);
   }
   //append centroid activations to activations from 0th quadrant.
   nClasses = dataset.getNumberClasses();   //normalize
   

   //reserve space for results
   vector<double> results(nClasses, 0);
   
   double maxResult = -99999;
   unsigned int maxLabel=0;
   //get max-result label
   for(size_t cl = 0; cl < nClasses; ++cl){
      results[cl] = svms[cl].classifyClassic(dataActivation, classicTrainActivations, &dataset);
      
      if(printResults)
         cout << "SVM " << cl << " says " << results[cl] << endl;  
      if(results[cl] > maxResult){
         maxResult = results[cl];

         maxLabel = cl;
      }
   }
   return maxLabel;
}



void CSVMClassifier::trainLinearNetwork(){
   unsigned int nTrainImages = dataset.getTrainSize();
   vector < vector < double > > datasetActivations;

   //allocate space for more vectors
   datasetActivations.reserve(nTrainImages);
   //for all trainings imagages:
   for(size_t dataIdx = 0; dataIdx < nTrainImages; ++dataIdx){
      vector < Feature > dataFeatures;
      //extract patches
      if(settings.codebook == CB_CODEBOOK){
        datasetActivations.push_back(getActivationsFromImage(dataset.getTrainImagePtr(dataIdx)));
     }else{
        datasetActivations.push_back(deepCodebook->getActivations(dataset.getTrainImagePtr(dataIdx)));
     }
   }
   //train the Linear Netwok with the gained activations
   linNetwork.train(datasetActivations, &dataset);
}



unsigned int CSVMClassifier::lnClassify(Image* image){
   vector<Patch> patches;
   vector<Feature> dataFeatures;
   vector<double> dataActivation;
   //extract patches
  
   if(settings.codebook == CB_CODEBOOK){
      dataActivation = getActivationsFromImage(image);
   }else{
      dataActivation = deepCodebook->getActivations(image);
   }
  
   return linNetwork.classify(dataActivation);

}
