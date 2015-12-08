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
   
   settings.netSettings.nCentroids = settings.codebookSettings.numberVisualWords;
   standardizeActivations = settings.codebookSettings.standardizeActivations;
   linNetwork.setSettings(settings.netSettings);
   convSVM.setSettings(settings.convSVMSettings);
}

void CSVMClassifier::train(){
   switch(settings.classifier){
      case CL_SVM:
         cout << "Training SVM..\n";
         trainClassicSVMs();
         
         break;
      case CL_CSVM:
         cout << "Training Conv SVM..\n";
         trainConvSVMs();
         
         break;
      case CL_LINNET:
         cout << "Training LinNet..\n";
         trainLinearNetwork();
         
         break;
      default:
         cout << "WARNING! couldnt recognize selected classifier!\n";
   }
}

unsigned int CSVMClassifier::classify(Image* im){
   unsigned int result = 0;
   
   switch(settings.classifier){
      case CL_SVM:
         result = classifyClassicSVMs(im, false); //return value should be processed
         break;
      case CL_CSVM:
         classifyConvSVM(im);
         break;
      case CL_LINNET:
         lnClassify(im);
         break;
   }
   return result;
}

//export the current codebook
void CSVMClassifier::exportCodebook(string filename){
   codebook.exportCodebook(filename);
}
//import the current codebook
void CSVMClassifier::importCodebook(string filename){
   codebook.importCodebook(filename);
}

/*
void CSVMClassifier::constructDeepCodebook(){
   //Construct patch layer
   unsigned int nPatches = settings.scannerSettings.nRandomPatches;
   vector<Feature> pretrainDump;
   
   cout << "constructing codebooks with " << settings.codebookSettings.numberVisualWords << " centroids, with " << nPatches << " patches\n";

   for(size_t pIdx = 0; pIdx < nPatches; ++pIdx){
      
      //patches = imageScanner.getRandomPatches(dataset.getImagePtrFromClass(im, cl));
      Patch patch = imageScanner.getRandomPatch(dataset.getImagePtr(rand() % dataset.getSize()));
      Feature newFeat = featExtr.extract(patch);
      pretrainDump.push_back(newFeat);//insert(pretrainDump[cl].end(),features.begin(),features.end());

   }

   cout << "Collected " << pretrainDump.size()<< " features\n";
   deepCodebook.constructPatchLayer(pretrainDump);
   
   cout << "done constructing patchLayer, using "  << settings.scannerSettings.nRandomPatches << " patches\n";
   pretrainDump.clear();
   
   //collect image features
   unsigned int nData = dataset.getSize();
   vector< vector<Patch> > imagePatches;
   vector< vector< vector<Feature> > > imageFeatures(nData, vector< vector<Feature> >(4));

   
   for(size_t imIdx = 0; imIdx < nData;  ++imIdx){
      imagePatches = imageScanner.scanImage(dataset.getImagePtr(imIdx));
      for(size_t quadIdx = 0; quadIdx < nData; ++quadIdx){
         unsigned int nFeatures = imagePatches[quadIdx].size();
         for(size_t pIdx = 0;  pIdx <nFeatures; ++pIdx){
            imageFeatures[imIdx][quadIdx][pIdx] = featExtr.extract(imagePatches[quadIdx][pIdx]);
         }
      }
   }
   //build hidden layers
   
   deepCodebook.constructHiddenLayers(imageFeatures);
}
*/


//return number of classes
unsigned int CSVMClassifier::getNoClasses(){
   return dataset.getNumberClasses();

}

//construct a codebook using the current dataset
void CSVMClassifier::constructCodebook(){
   
   unsigned int nClasses = dataset.getNumberClasses();
   
   unsigned int nPatches = settings.scannerSettings.nRandomPatches;
   
   bool oneCl = !settings.codebookSettings.useDifferentCodebooksPerClass;
   vector< vector<Feature> > pretrainDump(nClasses);
   cout << "constructing codebooks with " << settings.codebookSettings.numberVisualWords << " centroids for " << nClasses << " classes, with " << nPatches << " patches\n";
   
   
   for(size_t cl = 0; oneCl ? cl < 1 : cl < nClasses; ++cl){
      //cout << "parsing class " << cl << endl;
      
      //cout << nImages << " images\n";
      
      //cout << "space reserved\n";
      //cout << "Scanning " << nImages << " images\n";

      for(size_t pIdx = 0; pIdx < nPatches; ++pIdx){
         
         //patches = imageScanner.getRandomPatches(dataset.getImagePtrFromClass(im, cl));
         Patch patch = imageScanner.getRandomPatch(dataset.getImagePtr(rand() % dataset.getSize()));
         Feature newFeat = featExtr.extract(patch);
         pretrainDump[cl].push_back(newFeat);//insert(pretrainDump[cl].end(),features.begin(),features.end());

         
      }

      cout << "Collected " << pretrainDump[0].size()<< " features\n";
      codebook.constructCodebook(pretrainDump[cl],cl);
      
      cout << "done constructing codebook for class " << cl << " using "  << settings.scannerSettings.nRandomPatches << " patches\n";
   }
   pretrainDump.clear();
}

void CSVMClassifier::trainConvSVMs(){
      unsigned int datasetSize = dataset.getSize();
   vector < vector < vector < double > > > datasetActivations;
   vector < Feature > dataFeatures;
   vector < Patch > patches;
   vector< vector<double> > dataActivation;
   bool oneCl = !settings.codebookSettings.useDifferentCodebooksPerClass;
   //allocate space for more vectors
   datasetActivations.reserve(datasetSize);
   //for all trainings imagages:
   for(size_t dataIdx = 0; dataIdx < datasetSize; ++dataIdx){
      //cout << "scanning image" << dataIdx << endl;
      //extract patches
      patches = imageScanner.scanImage(dataset.getImagePtr(dataIdx));
      //cout << "for each quadrant..\n";
      
      //clear previous features
      dataFeatures.clear();
      //allocate for new features
      dataFeatures.reserve(patches.size());
      //cout << "extract features..\n";
      //extract features from all patches
      for(size_t patch = 0; patch < patches.size(); ++patch)
         dataFeatures.push_back(featExtr.extract(patches[patch]));
      patches.clear();
      //cout << "get activations..\n";
      //get cluster activations for the features
      dataActivation = codebook.getActivations(dataFeatures); 
      //cout << "dataActivation.size() = " << dataActivation.size() << endl;
      
      //append centroid activations to activations from 0th quadrant
      //cout << "appended activations\n";
      
      //normalize summed activations of appended pools
      unsigned int nClasses = dataActivation.size();
      
      //standardize data
      if(standardizeActivations){
         //cout << "nActivations = " << nActivations << endl;
         //cout << "***************** begin image ***********************\n";
         for(size_t clIdx = 0; oneCl ? clIdx < 1 : clIdx < nClasses; ++clIdx){
            double mean = 0;
            double stddev = 0;
            double nActivations = dataActivation[0].size();
            for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
               mean += dataActivation[clIdx][actIdx];
               //cout << "activation = " << dataActivation[0][0][actIdx] << endl;
            }
            mean /= nActivations;
            for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
               stddev+= (mean - dataActivation[clIdx][actIdx]) * (mean - dataActivation[clIdx][actIdx]);
            }
            stddev /= nActivations;
            stddev = sqrt(stddev + 0.01);
            for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
               //cout << "activ = " << dataActivation[0][0][actIdx] << endl;
               dataActivation[clIdx][actIdx] = (dataActivation[clIdx][actIdx] - mean)/stddev;
            }
            
         }
      }
      datasetActivations.push_back(dataActivation);
      
   }
   //cout << "Done getting activations\n";
   //train the Linear Netwok with the gained activations
   convSVM.train(datasetActivations, &dataset);
   
}

unsigned int CSVMClassifier::classifyConvSVM(Image* image){
      //cout << "nClasses = " << nClasses << endl;
   vector<Patch> patches;
   bool oneCl = !settings.codebookSettings.useDifferentCodebooksPerClass;
   vector<Feature> dataFeatures;
   vector< vector<double> > dataActivation;
   //extract patches
   patches = imageScanner.scanImage(image);
      
   
   //clear previous features
   dataFeatures.clear();
   //allocate for new features
   dataFeatures.reserve(patches.size());
   
   //extract features from all patches
   for(size_t patch = 0; patch < patches.size(); ++patch)
      dataFeatures.push_back(featExtr.extract(patches[patch]));
   patches.clear();
   
   //get cluster activations for the features
   dataActivation = codebook.getActivations(dataFeatures); 
   
   //append centroid activations to activations from 0th quadrant
   unsigned int nClasses = dataActivation[0].size();
   
   if(standardizeActivations){
      //cout << "nActivations = " << nActivations << endl;
      //cout << "***************** begin image ***********************\n";
      for(size_t clIdx = 0; oneCl ? clIdx < 1 : clIdx < nClasses; ++clIdx){
         double mean = 0;
         double stddev = 0;
         double nActivations = dataActivation[0].size();
         for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
            mean += dataActivation[clIdx][actIdx];
            //cout << "activation = " << dataActivation[0][0][actIdx] << endl;
         }
         mean /= nActivations;
         for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
            stddev+= (mean - dataActivation[clIdx][actIdx]) * (mean - dataActivation[clIdx][actIdx]);
         }
         stddev /= nActivations;
         stddev = sqrt(stddev + 0.01);
         for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
            //cout << "activ = " << dataActivation[0][0][actIdx] << endl;
            dataActivation[clIdx][actIdx] = (dataActivation[clIdx][actIdx] - mean)/stddev;
         }
         
      }
   }
   //cout << "*************\n";

   return convSVM.classify(dataActivation);
}
//train the KKT-SVM
void CSVMClassifier::trainClassicSVMs(){
   cout << "Enteing classic svm training\n";
   unsigned int datasetSize = dataset.getSize();
   unsigned int nClasses = dataset.getNumberClasses(); 
   unsigned int nCentroids; 
   
   vector< vector < vector<double> > > datasetActivations;
   vector < Feature > dataFeatures;
   vector < Patch > patches;

   vector < vector<double> > dataKernel(datasetSize);
   vector< vector<double> > dataActivation;

   bool oneCl = !settings.codebookSettings.useDifferentCodebooksPerClass;
   //allocate space for more vectors
   datasetActivations.reserve(datasetSize);
   cout << "collecting activations for trainingsdata..\n";
   //for all trainings imagages:
   for(size_t dataIdx = 0; dataIdx < datasetSize; ++dataIdx){
      
      //extract patches
      patches = imageScanner.scanImage(dataset.getImagePtr(dataIdx));
      dataActivation.clear();
      dataFeatures.clear();
      //clear previous features
      
      //allocate for new features
      dataFeatures.reserve(patches.size());
      
      //cout << patches[qIdx].size() << " patches" << endl;
      //extract features from all patches
      for(size_t patch = 0; patch < patches.size(); ++patch)
         dataFeatures.push_back(featExtr.extract(patches[patch]));
      patches.clear();
      
      //get cluster activations for the features
      dataActivation = codebook.getActivations(dataFeatures); 
      dataFeatures.clear();
   
      patches.clear();
      //append centroid activations to activations from 0th quadrant
      //normalize data
      
      if(standardizeActivations){
         //cout << "nActivations = " << nActivations << endl;
         //cout << "***************** begin image ***********************\n";
         for(size_t clIdx = 0; oneCl ? clIdx < 1 : clIdx < nClasses; ++clIdx){
            double mean = 0;
            double stddev = 0;
            double nActivations = dataActivation[0].size();
            for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
               mean += dataActivation[clIdx][actIdx];
               //cout << "activation = " << dataActivation[0][0][actIdx] << endl;
            }
            mean /= nActivations;
            for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
               stddev+= (mean - dataActivation[clIdx][actIdx]) * (mean - dataActivation[clIdx][actIdx]);
            }
            stddev /= nActivations;
            stddev = sqrt(stddev + 0.01);
            for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
               //cout << "activ = " << dataActivation[0][0][actIdx] << endl;
               dataActivation[clIdx][actIdx] = (dataActivation[clIdx][actIdx] - mean)/stddev;
            }
            
         }
      }
      //get cluster activations for the features
     datasetActivations.push_back(dataActivation);
     dataActivation.clear();
   }
   nClasses = dataset.getNumberClasses();
   nCentroids = datasetActivations[0].size();
   
   
   
   cout << "Calculating similarities\n";
   //calculate similarity kernal between activation vectors
   for(size_t dIdx0 = 0; dIdx0 < datasetSize; ++dIdx0){
      //cout << "done with similarity of " << dIdx0 << endl;
      for(size_t dIdx1 = 0; dIdx1 <= dIdx0; ++dIdx1){
         dataKernel[dIdx0].resize(dIdx0 + 1);
         double sum = 0;
         if(settings.svmSettings.kernelType == RBF){
            

            for(size_t cl = 0;   oneCl ? cl < 1 : cl < nClasses; ++cl){

               for(size_t centr = 0; centr < nCentroids; ++centr){
                  sum += (datasetActivations[dIdx0][cl][centr] - datasetActivations[dIdx1][cl][centr])*(datasetActivations[dIdx0][cl][centr] - datasetActivations[dIdx1][cl][centr]);
               }
            }
            dataKernel[dIdx0][dIdx1] = exp((-1.0 * sum)/settings.svmSettings.sigmaClassicSimilarity);
            //dataKernel[dIdx1][dIdx0] = dataKernel[dIdx0][dIdx1];
            
         }else if (settings.svmSettings.kernelType == LINEAR){

            for(size_t cl = 0;   oneCl ? cl < 1 : cl < nClasses; ++cl){

               for(size_t centr = 0; centr < nCentroids; ++centr){
                  sum += (datasetActivations[dIdx0][cl][centr] * datasetActivations[dIdx1][cl][centr]);
               }
            }
            //cout << "Writing " << sum << " to " << dIdx0 << ", " << dIdx1 << endl;
            dataKernel[dIdx0][dIdx1] = sum;
            //dataKernel[dIdx1][dIdx0] = sum;
         }else
            cout << "CSVM::svm::Error! No valid kernel type selected! Try: RBF or LINEAR\n"  ;
         
      }
   }
   //print part of the sim kernel for debugging purposes
   /*for(size_t dIdx0 = 0; dIdx0 < 14; ++dIdx0){
      for(size_t dIdx1 = 0; dIdx1 < 14; ++dIdx1){
         cout << (dIdx0 == dIdx1 ? "*": "") << (dataset.getImagePtr(dIdx0)->getLabelId()==dataset.getImagePtr(dIdx1)->getLabelId() ? "!" : "") << "(" << dataset.getImagePtr(dIdx0)->getLabelId() << ", " << dataset.getImagePtr(dIdx1)->getLabelId() << ")" << setprecision(2) << dataKernel[dIdx0][dIdx1] << ",\t";
      } 
      cout << endl;
      cout << setprecision(5) ;
   }*/
   //we have a similarity kernel, now train the SVM's
   
   for(size_t cl = 0; cl < nClasses; ++cl){
      svms[cl].trainClassic(dataKernel, &dataset);  
   }
   classicTrainActivations = datasetActivations;
}

//train the convolutional SVMs
void CSVMClassifier::trainSVMs(){
   unsigned int datasetSize = dataset.getSize();
   vector < vector < vector < double > > > datasetActivations;
   vector < Feature > dataFeatures;
   vector < Patch > patches;
   
   vector < vector<double> > dataActivation;
   //allocate space for more vectors
   datasetActivations.reserve(datasetSize);
   //for all trainings imagages:
   for(size_t dataIdx = 0; dataIdx < datasetSize; ++dataIdx){
      
      //extract patches
      patches = imageScanner.scanImage(dataset.getImagePtr(dataIdx));
      
      //clear previous features
      dataFeatures.clear();
      //allocate for new features
      dataFeatures.reserve(patches.size());
      
      //extract features from all patches
      for(size_t patch = 0; patch < patches.size(); ++patch)
         dataFeatures.push_back(featExtr.extract(patches[patch]));
      patches.clear();
      
      //get cluster activations for the features
      dataActivation = codebook.getActivations(dataFeatures); 
      
      unsigned int nClasses = dataset.getNumberClasses();
      //append centroid activations to activations from 0th quadrant
      
      if(standardizeActivations){
         //cout << "nActivations = " << nActivations << endl;
         //cout << "***************** begin image ***********************\n";
         for(size_t clIdx = 0; clIdx < nClasses; ++clIdx){
            double mean = 0;
            double stddev = 0;
            double nActivations = dataActivation[0].size();
            for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
               mean += dataActivation[clIdx][actIdx];
               //cout << "activation = " << dataActivation[0][0][actIdx] << endl;
            }
            mean /= nActivations;
            for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
               stddev+= (mean - dataActivation[clIdx][actIdx]) * (mean - dataActivation[clIdx][actIdx]);
            }
            stddev /= nActivations;
            stddev = sqrt(stddev + 0.01);
            for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
               //cout << "activ = " << dataActivation[0][0][actIdx] << endl;
               dataActivation[clIdx][actIdx] = (dataActivation[clIdx][actIdx] - mean)/stddev;
            }
            
         }
      }
      datasetActivations.push_back(dataActivation);
   }
   //cout << "Done getting activations\n";
   //train the SVMs with the gained activations
   for(size_t svmIdx = 0; svmIdx < svms.size(); ++svmIdx){
      svms[svmIdx].train(datasetActivations, &dataset);
   }
}

//classify an image using the convolutional SVMs
/*unsigned int CSVMClassifier::classify(Image* image){
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
   for(size_t qIdx = 1; qIdx < 4; ++qIdx){
      for(size_t clIdx = 0; clIdx < nClasses; ++clIdx){
         dataActivation[0][clIdx].insert(dataActivation[0][clIdx].end(),dataActivation[qIdx][clIdx].begin(), dataActivation[qIdx][clIdx].end());
         dataActivation[qIdx][clIdx].clear();
      }
   }
   
   
   
   if(standardizeActivations){
      //cout << "nActivations = " << nActivations << endl;
      //cout << "***************** begin image ***********************\n";
      for(size_t clIdx = 0; clIdx < nClasses; ++clIdx){
         double mean = 0;
         double stddev = 0;
         double nActivations = dataActivation[0][0].size();
         for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
            mean += dataActivation[0][clIdx][actIdx];
            //cout << "activation = " << dataActivation[0][0][actIdx] << endl;
         }
         mean /= nActivations;
         for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
            stddev+= (mean - dataActivation[0][clIdx][actIdx]) * (mean - dataActivation[0][clIdx][actIdx]);
         }
         stddev /= nActivations;
         stddev = sqrt(stddev + 0.01);
         for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
            //cout << "activ = " << dataActivation[0][0][actIdx] << endl;
            dataActivation[0][clIdx][actIdx] = (dataActivation[0][clIdx][actIdx] - mean)/stddev;
         }
         
      }
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
}*/

//classify an image using the KKT-SVM
unsigned int CSVMClassifier::classifyClassicSVMs(Image* image, bool printResults){
   unsigned int nClasses = dataset.getNumberClasses();
   //cout << "nClasses = " << nClasses << endl;
   vector<Patch> patches;
   vector<Feature> dataFeatures;
   bool oneCl = !settings.codebookSettings.useDifferentCodebooksPerClass;
   vector< vector<double> > dataActivation;
   //extract patches
   patches = imageScanner.scanImage(image);
      
   //clear previous features
   dataFeatures.clear();
   //allocate for new features
   dataFeatures.reserve(patches.size());
   
   //extract features from all patches
   for(size_t patch = 0; patch < patches.size(); ++patch)
      dataFeatures.push_back(featExtr.extract(patches[patch]));
   patches.clear();
   
   //get cluster activations for the features
   dataActivation = codebook.getActivations(dataFeatures); 
   
   //append centroid activations to activations from 0th quadrant.
   nClasses = dataset.getNumberClasses();   //normalize
   //cout << "Normalizing data" << endl;
   if(standardizeActivations){
      //cout << "nActivations = " << nActivations << endl;
      //cout << "***************** begin image ***********************\n";
      for(size_t clIdx = 0; oneCl ? clIdx < 1 : clIdx < nClasses; ++clIdx){
         double mean = 0;
         double stddev = 0;
         double nActivations = dataActivation[0].size();
         for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
            mean += dataActivation[clIdx][actIdx];
            //cout << "activation = " << dataActivation[0][0][actIdx] << endl;
         }
         mean /= nActivations;
         for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
            stddev+= (mean - dataActivation[clIdx][actIdx]) * (mean - dataActivation[clIdx][actIdx]);
         }
         stddev /= nActivations;
         stddev = sqrt(stddev + 0.01);
         for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
            //cout << "activ = " << dataActivation[0][0][actIdx] << endl;
            dataActivation[clIdx][actIdx] = (dataActivation[clIdx][actIdx] - mean)/stddev;
         }
         
      }
   }
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

bool CSVMClassifier::useClassicSVM(){
   return settings.svmSettings.type == CLASSIC;
}

void CSVMClassifier::trainLinearNetwork(){
   unsigned int datasetSize = dataset.getSize();
   vector < vector < vector < double > > > datasetActivations;
   vector < Feature > dataFeatures;
   vector < Patch > patches;
   vector< vector<double> > dataActivation;
   bool oneCl = !settings.codebookSettings.useDifferentCodebooksPerClass;
   //allocate space for more vectors
   datasetActivations.reserve(datasetSize);
   //for all trainings imagages:
   for(size_t dataIdx = 0; dataIdx < datasetSize; ++dataIdx){
      //cout << "scanning image" << dataIdx << endl;
      //extract patches
      patches = imageScanner.scanImage(dataset.getImagePtr(dataIdx));
      //cout << "for each quadrant..\n";
      //clear previous features
      dataFeatures.clear();
      //allocate for new features
      dataFeatures.reserve(patches.size());
      //cout << "extract features..\n";
      //extract features from all patches
      for(size_t patch = 0; patch < patches.size(); ++patch)
         dataFeatures.push_back(featExtr.extract(patches[patch]));
      patches.clear();
      //cout << "get activations..\n";
      //get cluster activations for the features
      dataActivation = codebook.getActivations(dataFeatures); 
      //cout << "dataActivation.size() = " << dataActivation.size() << endl;
   
      //append centroid activations to activations from 0th quadrant
      //cout << "appended activations\n";
      
      //normalize summed activations of appended pools
      unsigned int nClasses = dataActivation[0].size();
      
      //standardize data
      if(standardizeActivations){
         //cout << "nActivations = " << nActivations << endl;
         //cout << "***************** begin image ***********************\n";
         for(size_t clIdx = 0; oneCl ? clIdx < 1 : clIdx < nClasses; ++clIdx){
            double mean = 0;
            double stddev = 0;
            double nActivations = dataActivation[0].size();
            for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
               mean += dataActivation[clIdx][actIdx];
               //cout << "activation = " << dataActivation[0][0][actIdx] << endl;
            }
            mean /= nActivations;
            for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
               stddev+= (mean - dataActivation[clIdx][actIdx]) * (mean - dataActivation[clIdx][actIdx]);
            }
            stddev /= nActivations;
            stddev = sqrt(stddev + 0.01);
            for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
               //cout << "activ = " << dataActivation[0][0][actIdx] << endl;
               dataActivation[clIdx][actIdx] = (dataActivation[clIdx][actIdx] - mean)/stddev;
            }
            
         }
      }
      datasetActivations.push_back(dataActivation);
      
   }
   //cout << "Done getting activations\n";
   //train the Linear Netwok with the gained activations
   linNetwork.train(datasetActivations, &dataset);
}

unsigned int CSVMClassifier::lnClassify(Image* image){
   //cout << "nClasses = " << nClasses << endl;
   vector<Patch> patches;
   bool oneCl = !settings.codebookSettings.useDifferentCodebooksPerClass;
   vector<Feature> dataFeatures;
   vector< vector<double> > dataActivation;
   //extract patches
   patches = imageScanner.scanImage(image);
      
   //clear previous features
   dataFeatures.clear();
   //allocate for new features
   dataFeatures.reserve(patches.size());
   
   //extract features from all patches
   for(size_t patch = 0; patch < patches.size(); ++patch)
      dataFeatures.push_back(featExtr.extract(patches[patch]));
   patches.clear();
   
   //get cluster activations for the features
   dataActivation = codebook.getActivations(dataFeatures); 
   
   //append centroid activations to activations from 0th quadrant
   unsigned int nClasses = dataActivation.size();
   
   if(standardizeActivations){
      //cout << "nActivations = " << nActivations << endl;
      //cout << "***************** begin image ***********************\n";
      for(size_t clIdx = 0; oneCl ? clIdx < 1 : clIdx < nClasses; ++clIdx){
         double mean = 0;
         double stddev = 0;
         double nActivations = dataActivation[0].size();
         for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
            mean += dataActivation[clIdx][actIdx];
            //cout << "activation = " << dataActivation[0][0][actIdx] << endl;
         }
         mean /= nActivations;
         for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
            stddev+= (mean - dataActivation[clIdx][actIdx]) * (mean - dataActivation[clIdx][actIdx]);
         }
         stddev /= nActivations;
         stddev = sqrt(stddev + 0.01);
         for(size_t actIdx = 0; actIdx < nActivations; ++actIdx){
            //cout << "activ = " << dataActivation[0][0][actIdx] << endl;
            dataActivation[clIdx][actIdx] = (dataActivation[clIdx][actIdx] - mean)/stddev;
         }
         
      }
   }
   //cout << "*************\n";

   return linNetwork.classify(dataActivation);

   //return labelId of max-output-SVM
 //  return maxLabel;
}
