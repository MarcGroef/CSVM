#include <csvm/csvm_classifier.h>
#include <iomanip>
//
using namespace std;
using namespace csvm;

//initialize random
CSVMClassifier::CSVMClassifier(){
   srand(time(NULL)); 
   
}


//initialize the SVMs, by telling them the dataset size, amount of classes, centroids, and the respective label of the SVM
void CSVMClassifier::initSVMs(){
   svms.reserve(codebook.getNClasses());
   for(size_t svmIdx = 0; svmIdx < codebook.getNClasses(); ++svmIdx){
      svms.push_back(SVM(dataset.getSize(), codebook.getNClasses(), codebook.getNCentroids(), svmIdx));
      svms[svmIdx].setSettings(settings.svmSettings);
      //cout << "I gave the SVMS alphainit = " << settings.svmSettings.alphaDataInit << endl;
   }
}

//read settings file, and pass the settings to respective modules
void CSVMClassifier::setSettings(string settingsFile){
   settings.readSettingsFile(settingsFile);
   //analyser.setSettings(settings.analyserSettings);
   imageScanner.setSettings(settings.scannerSettings);
   dataset.setSettings(settings.datasetSettings);
   codebook.setSettings(settings.codebookSettings);
   featExtr.setSettings(settings.featureSettings);
}

//export the current codebook
void CSVMClassifier::exportCodebook(string filename){
   codebook.exportCodebook(filename);
}
//import the current codebook
void CSVMClassifier::importCodebook(string filename){
   codebook.importCodebook(filename);
}

//construct a codebook using the current dataset
void CSVMClassifier::constructCodebook(){
   
   unsigned int nClasses = dataset.getNumberClasses();
   
   bool allocedDump = false;
   unsigned int nPatches = settings.scannerSettings.nRandomPatches;
   vector<Patch> patches;
   vector<Feature> features;
   
   features.reserve(settings.scannerSettings.nRandomPatches);
   
   pretrainDump.clear();
   pretrainDump.resize(nClasses);
   //unsigned int nImages = dataset.getSize();
   
   unsigned int nImages = 10000;
   //cout << "constructing codebooks with " << settings.codebookSettings.numberVisualWords << " centroids for " << nClasses << " classes using " << nImages << " images in total\n";
   bool oneCl = settings.svmSettings.useDifferentCodebooksPerClass;
   for(size_t cl = 0;  oneCl ? cl < 1 : cl < nClasses; ++cl){
      allocedDump = false;
      //cout << "parsing class " << cl << endl;
      if(oneCl){
         nImages = dataset.getNumberImagesInClass(cl);
      }
      //cout << nImages << " images\n";
      
      
      //cout << "space reserved\n";
      //cout << "Scanning " << nImages << " images\n";
      for(size_t im = 0; im < nImages; ++im){
         //cout << "scanning patches\n";
         if(oneCl){
            
            patches = imageScanner.getRandomPatches(dataset.getImagePtrFromClass(rand() & nImages, cl));
            
         }else
            patches = imageScanner.getRandomPatches(dataset.getImagePtr(rand() % dataset.getSize()));
         
         nPatches = patches.size();
         //cout << "Extracted " << nPatches << " patches\n";
         //checkEqualPatches(patches);
         features.clear();
         features.reserve(nPatches);
         //cout << "extracting patches..\n";
         for(size_t patchIdx = 0; patchIdx < nPatches; ++patchIdx){
            Feature newFeat = featExtr.extract(patches[patchIdx]);
            features.push_back(newFeat);
            //cout << "first element from new Feature = " << newFeat.content[0] << ", which should equal " << features[patchIdx].content[0] << endl;
         }
         if(!allocedDump){
            pretrainDump[cl].reserve(nPatches * nImages);
            allocedDump = true;
         }
         //cout << "extracted " << features[0].content.size() << "x1 features\n";
         //checkEqualFeatures(features);
         pretrainDump[cl].insert(pretrainDump[cl].end(),features.begin(),features.end());
         
      }
      //checkEqualFeatures(pretrainDump[cl]);
      //cout << "end feature meditation\n";
      //cout << pretrainDump[cl].size() << " features extracted for class " << cl << "\n";
      //checkEqualFeatures(pretrainDump[cl]);
      cout << "Collected features\n";
      codebook.constructCodebook(pretrainDump[cl],cl);
      //checkEqualFeatures(pretrainDump[cl]);
      cout << "done constructing codebook for class " << cl << " using " << nImages << " images, " << settings.scannerSettings.nRandomPatches << " patches each: " << settings.scannerSettings.nRandomPatches * nImages<<" in total.\n";
   }
   pretrainDump.clear();
}

//train the KKT-SVM
vector < vector< vector<double> > > CSVMClassifier::trainClassicSVMs(){
   unsigned int datasetSize = dataset.getSize();
   unsigned int nClasses; 
   unsigned int nCentroids; 
   
   vector < vector < vector<double> > > datasetActivations;
   vector < Feature > dataFeatures;
   vector< vector < Patch > > patches(4);
   vector < vector<double> > dataKernel(datasetSize, vector<double>(datasetSize,0.0));
   vector < vector< vector<double> > > dataActivation;

   
   //allocate space for more vectors
   datasetActivations.reserve(datasetSize);
   //cout << "collecting activations for trainingsdata..\n";
   //for all trainings imagages:
   for(size_t dataIdx = 0; dataIdx < datasetSize; ++dataIdx){
      
      //extract patches
      patches = imageScanner.scanImage(dataset.getImagePtr(dataIdx));
      
      for(size_t qIdx = 0; qIdx < 4; ++qIdx){
         //clear previous features
         dataFeatures.clear();
         //allocate for new features
         dataFeatures.reserve(patches[qIdx].size());
         
         //extract features from all patches
         for(size_t patch = 0; patch < patches[qIdx].size(); ++patch)
            dataFeatures.push_back(featExtr.extract(patches[qIdx][patch]));
         patches[qIdx].clear();
         
         //get cluster activations for the features
         dataActivation.push_back(codebook.getActivations(dataFeatures)); 
         
      }
      //append centroid activations to activations from 0th quadrant
      unsigned int nCentroids = dataActivation[0][0].size();
      for(size_t qIdx = 1; qIdx < 4; ++qIdx){
         dataActivation[0][0].insert(dataActivation[0][0].end(),dataActivation[qIdx][0].begin(), dataActivation[qIdx][0].end());
      }
      
      //get cluster activations for the features
     datasetActivations.push_back(dataActivation[0]);
   }
   
   nClasses = datasetActivations[0].size();
   nCentroids = datasetActivations[0][0].size();
   
   
   
   
   bool oneCl = settings.svmSettings.useDifferentCodebooksPerClass;
   //calculate similarity kernal between activation vectors
   for(size_t dIdx0 = 0; dIdx0 < datasetSize; ++dIdx0){
      //cout << "done with similarity of " << dIdx0 << endl;
      for(size_t dIdx1 = dIdx0; dIdx1 < datasetSize; ++dIdx1){
         double sum = 0;
         if(settings.svmSettings.kernelType == RBF){
            
            for(size_t cl = 0;   oneCl ? cl < 1 : cl < nClasses; ++cl){
               for(size_t centr = 0; centr < nCentroids; ++centr){
                  sum += (datasetActivations[dIdx0][cl][centr] - datasetActivations[dIdx1][cl][centr])*(datasetActivations[dIdx0][cl][centr] - datasetActivations[dIdx1][cl][centr]);
               }
            }
            dataKernel[dIdx0][dIdx1] = exp((-1.0 * sum)/settings.svmSettings.sigmaClassicSimilarity);
            dataKernel[dIdx1][dIdx0] = dataKernel[dIdx0][dIdx1];
            
         }else if (settings.svmSettings.kernelType == LINEAR){
            for(size_t cl = 0;   oneCl ? cl < 1 : cl < nClasses; ++cl){
               for(size_t centr = 0; centr < nCentroids; ++centr){
                  sum += (datasetActivations[dIdx0][cl][centr] * datasetActivations[dIdx1][cl][centr]);
               }
            }
            dataKernel[dIdx0][dIdx1] = sum;
            dataKernel[dIdx1][dIdx0] = sum;
         }else
            cout << "CSVM::svm::Error! No valid kernel type selected! Try: RBF or LINEAR\n"  ;
         
      }
   }
   //print part of the sim kernel for debugging purposes
   /*for(size_t dIdx0 = 0; dIdx0 < 14; ++dIdx0){
      for(size_t dIdx1 = 0; dIdx1 < 14; ++dIdx1){
         cout << (dIdx0 == dIdx1 ? "*": "") << (dataset.getImagePtr(dIdx0)->getLabelId()==dataset.getImagePtr(dIdx1)->getLabelId() ? "!" : "") << "(" << dataset.getImagePtr(dIdx0)->getLabelId() << ", " << dataset.getImagePtr(dIdx1)->getLabelId() << ")" << setprecision(2) << dataKernel[dIdx0].content[dIdx1] << ",\t";
      } 
      cout << endl;
      cout << setprecision(5) ;
   }*/
   //we have a similarity kernel, now train the SVM's
   
   for(size_t cl = 0; cl < nClasses; ++cl){
      svms[cl].trainClassic(dataKernel, &dataset);  
   }
   return datasetActivations;
}

//train the convolutional SVMs
void CSVMClassifier::trainSVMs(){
   unsigned int datasetSize = dataset.getSize();
   vector < vector < vector < double > > > datasetActivations;
   vector < Feature > dataFeatures;
   vector< vector < Patch > > patches(4);
   
   vector < vector< vector<double> > > dataActivation;
   //allocate space for more vectors
   datasetActivations.reserve(datasetSize);
   //for all trainings imagages:
   for(size_t dataIdx = 0; dataIdx < datasetSize; ++dataIdx){
      
      //extract patches
      patches = imageScanner.scanImage(dataset.getImagePtr(dataIdx));
      
      for(size_t qIdx = 0; qIdx < 4; ++qIdx){
         //clear previous features
         dataFeatures.clear();
         //allocate for new features
         dataFeatures.reserve(patches[qIdx].size());
         
         //extract features from all patches
         for(size_t patch = 0; patch < patches[qIdx].size(); ++patch)
            dataFeatures.push_back(featExtr.extract(patches[qIdx][patch]));
         patches[qIdx].clear();
         
         //get cluster activations for the features
         dataActivation.push_back(codebook.getActivations(dataFeatures)); 
      }
      //append centroid activations to activations from 0th quadrant
      for(size_t qIdx = 1; qIdx < 4; ++qIdx){
         dataActivation[0][0].insert(dataActivation[0][0].end(),dataActivation[qIdx][0].begin(), dataActivation[qIdx][0].end());
      }
      datasetActivations.push_back(dataActivation[0]);
   }
   //cout << "Done getting activations\n";
   //train the SVMs with the gained activations
   for(size_t svmIdx = 0; svmIdx < svms.size(); ++svmIdx){
      svms[svmIdx].train(datasetActivations, &dataset);
   }
}

//classify an image using the convolutional SVMs
unsigned int CSVMClassifier::classify(Image* image){
   unsigned int nClasses = codebook.getNClasses();
   //cout << "nClasses = " << nClasses << endl;
   vector< vector<Patch> > patches(4);
   vector<Feature> dataFeatures;
   
   vector < vector< vector<double> > > dataActivation;
   //extract patches
   patches = imageScanner.scanImage(image);
   

   //allocate for new
    patches = imageScanner.scanImage(image);
      
   for(size_t qIdx = 0; qIdx < 4; ++qIdx){
      //clear previous features
      dataFeatures.clear();
      //allocate for new features
      dataFeatures.reserve(patches[qIdx].size());
      
      //extract features from all patches
      for(size_t patch = 0; patch < patches[qIdx].size(); ++patch)
         dataFeatures.push_back(featExtr.extract(patches[qIdx][patch]));
      patches[qIdx].clear();
      
      //get cluster activations for the features
      dataActivation.push_back(codebook.getActivations(dataFeatures)); 
   }
   //append centroid activations to activations from 0th quadrant
   for(size_t qIdx = 1; qIdx < 4; ++qIdx){
         dataActivation[0][0].insert(dataActivation[0][0].end(),dataActivation[qIdx][0].begin(), dataActivation[qIdx][0].end());
   }
   
   patches.clear();

   //reserve space for all SVM results
   vector<double> results(nClasses, 0);
   double maxResult = -99999;
   unsigned int maxLabel=0;
   //get max results
   //cout << "*************\n";
   for(size_t cl = 0; cl < nClasses; ++cl){
      results[cl] = svms[cl].classify(dataActivation[0], &codebook);
      
      if(results[cl] > maxResult){
         maxResult = results[cl];

         maxLabel = cl;
      }
   }
   //return labelId of max-output-SVM
   return maxLabel;
}

//classify an image using the KKT-SVM
unsigned int CSVMClassifier::classifyClassicSVMs(Image* image, vector < vector< vector<double> > >& trainActivations, bool printResults){
   unsigned int nClasses = codebook.getNClasses();
   //cout << "nClasses = " << nClasses << endl;
   vector < vector<Patch> > patches(4);
   vector<Feature> dataFeatures;
   
   vector < vector< vector<double> > > dataActivation;
   //extract patches
   patches = imageScanner.scanImage(image);
      
   for(size_t qIdx = 0; qIdx < 4; ++qIdx){
      //clear previous features
      dataFeatures.clear();
      //allocate for new features
      dataFeatures.reserve(patches[qIdx].size());
      
      //extract features from all patches
      for(size_t patch = 0; patch < patches[qIdx].size(); ++patch)
         dataFeatures.push_back(featExtr.extract(patches[qIdx][patch]));
      patches[qIdx].clear();
      
      //get cluster activations for the features
      dataActivation.push_back(codebook.getActivations(dataFeatures)); 
   }
   //append centroid activations to activations from 0th quadrant.
   for(size_t qIdx = 1; qIdx < 4; ++qIdx){
      dataActivation[0][0].insert(dataActivation[0][0].end(),dataActivation[qIdx][0].begin(), dataActivation[qIdx][0].end());
      
   }
   
   //reserve space for results
   vector<double> results(nClasses, 0);
   
   double maxResult = -99999;
   unsigned int maxLabel=0;
   
   //get max-result label
   for(size_t cl = 0; cl < nClasses; ++cl){
      results[cl] = svms[cl].classifyClassic(dataActivation[0], trainActivations, &dataset);
      
      if(printResults)
         cout << "SVM " << cl << " says " << results[cl] << endl;  
      if(results[cl] > maxResult){
         maxResult = results[cl];

         maxLabel = cl;
      }
   }
   return maxLabel;
}

bool CSVMClassifier::useClassicSVM(){
   return settings.svmSettings.type == CLASSIC;
}

void CSVMClassifier::trainLinearNetwork(){
   unsigned int datasetSize = dataset.getSize();
   vector < vector < vector < double > > > datasetActivations;
   vector < Feature > dataFeatures;
   vector< vector < Patch > > patches(4);
   vector < vector< vector<double> > > dataActivation;
   //allocate space for more vectors
   datasetActivations.reserve(datasetSize);
   //for all trainings imagages:
   for(size_t dataIdx = 0; dataIdx < datasetSize; ++dataIdx){
      //cout << "scanning image" << dataIdx << endl;
      //extract patches
      patches = imageScanner.scanImage(dataset.getImagePtr(dataIdx));
      //cout << "for each quadrant..\n";
      for(size_t qIdx = 0; qIdx < 4; ++qIdx){
         //clear previous features
         dataFeatures.clear();
         //allocate for new features
         dataFeatures.reserve(patches[qIdx].size());
         //cout << "extract features..\n";
         //extract features from all patches
         for(size_t patch = 0; patch < patches[qIdx].size(); ++patch)
            dataFeatures.push_back(featExtr.extract(patches[qIdx][patch]));
         patches[qIdx].clear();
         //cout << "get activations..\n";
         //get cluster activations for the features
         dataActivation.push_back(codebook.getActivations(dataFeatures)); 
         //cout << "dataActivation.size() = " << dataActivation.size() << endl;
      }
      //append centroid activations to activations from 0th quadrant
      //cout << "appended activations\n";
      unsigned int nCentroids = dataActivation[0][0].size();
      for(size_t qIdx = 1; qIdx < 4; ++qIdx){
        // cout << "append qIdx = " << qIdx << endl;
         //cout << "activVector size = " << dataActivation[0][0].size();
         dataActivation[0][0].insert(dataActivation[0][0].end(),dataActivation[qIdx][0].begin(), dataActivation[qIdx][0].end());
      }
      datasetActivations.push_back(dataActivation[0]);
      
   }
   //cout << "Done getting activations\n";
   //train the Linear Netwok with the gained activations
   linNetwork.train(datasetActivations, &dataset);
}

unsigned int CSVMClassifier::lnClassify(Image* image){
   unsigned int nClasses = codebook.getNClasses();
   //cout << "nClasses = " << nClasses << endl;
   vector< vector<Patch> > patches(4);
   vector<Feature> dataFeatures;
   vector < vector< vector<double> > > dataActivation;
   //extract patches
  patches = imageScanner.scanImage(image);
      
   for(size_t qIdx = 0; qIdx < 4; ++qIdx){
      //clear previous features
      dataFeatures.clear();
      //allocate for new features
      dataFeatures.reserve(patches[qIdx].size());
      
      //extract features from all patches
      for(size_t patch = 0; patch < patches[qIdx].size(); ++patch)
         dataFeatures.push_back(featExtr.extract(patches[qIdx][patch]));
      patches[qIdx].clear();
      
      //get cluster activations for the features
      dataActivation.push_back(codebook.getActivations(dataFeatures)); 
   }
   //append centroid activations to activations from 0th quadrant
   for(size_t qIdx = 1; qIdx < 4; ++qIdx){
      dataActivation[0][0].insert(dataActivation[0][0].end(),dataActivation[qIdx][0].begin(), dataActivation[qIdx][0].end());
      
   }

   //cout << "*************\n";

   return linNetwork.classify(dataActivation[0]);

   //return labelId of max-output-SVM
 //  return maxLabel;
}