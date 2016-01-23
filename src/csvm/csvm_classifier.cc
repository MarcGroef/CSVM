#include <csvm/csvm_classifier.h>
#include <iomanip>
//
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

//initialize the SVMs, by telling them the dataset size, amount of classes, centroids, and the respective label of the SVM
void CSVMClassifier::initSVMs(){
   
   unsigned int ensembleSize = dataset.getNumberClasses();
   
   svms.reserve(dataset.getNumberClasses());
   for(size_t svmIdx = 0; svmIdx < ensembleSize; ++svmIdx){
      svms.push_back(SVM(dataset.getTrainSize(), codebook.getNClasses(), codebook.getNCentroids(), svmIdx));
      svms[svmIdx].setSettings(settings.svmSettings);
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
   linNetwork.setSettings(settings.netSettings);
   convSVM.setSettings(settings.convSVMSettings);
}


void CSVMClassifier::train(){
   switch(settings.classifier){
      case CL_SVM:
         //cout << "Training SVM..\n";
         trainClassicSVMs();
         
         break;
      case CL_CSVM:
         //cout << "Training Conv SVM..\n";
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
         result = classifyConvSVM(im);
         break;
      case CL_LINNET:
         result = lnClassify(im);
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


void CSVMClassifier::constructDeepCodebook(){
   deepCodebook->setSettings(settings.dcbSettings);
   deepCodebook->generateCentroids();
   //cout << "Done constructing deep codebook\n";
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
   //cout << "constructing codebooks with " << settings.codebookSettings.numberVisualWords << " centroids for " << nClasses << " classes, with " << nPatches << " patches\n";


   for(size_t pIdx = 0; pIdx < nPatches; ++pIdx){
      
      //patches = imageScanner.getRandomPatches(dataset.getImagePtrFromClass(im, cl));
      Patch patch = imageScanner.getRandomPatch(dataset.getImagePtr(rand() % dataset.getTotalImages()));
      Feature newFeat = featExtr.extract(patch);
      pretrainDump.push_back(newFeat);//insert(pretrainDump[cl].end(),features.begin(),features.end());

      
   }

   //cout << "Collected " << pretrainDump.size()<< " features\n";
   codebook.constructCodebook(pretrainDump);
   
   //cout << "done constructing codebook using "  << settings.scannerSettings.nRandomPatches << " patches\n";
   
   pretrainDump.clear();
}


void CSVMClassifier::trainConvSVMs(){
   int nTrainImages = dataset.getTrainSize();
   vector < vector < Feature > > dataFeaturesVec;
   vector < Feature > dataFeatures;
   vector < Patch > patches;
   dataFeaturesVec.reserve(nTrainImages);
   cout << "Extracting Features from trainingdata... " << fixed << setprecision(0) << endl;
   double percentage;
   //for all trainings imagages:
   for(int dataIdx = 0; dataIdx < nTrainImages; ++dataIdx){
      
      percentage = (double) dataIdx / nTrainImages * 100;
      cout << "\r " << percentage << " %          " << flush;
      if(settings.codebook == CB_CODEBOOK){
         //clear previous features
         patches = imageScanner.scanImage(dataset.getTrainImagePtr(dataIdx));
         dataFeatures.clear();
         
         //allocate for new features
         dataFeatures.reserve(patches.size());
         
         //extract features from all patches
         for(size_t patch = 0; patch < patches.size(); ++patch)
            dataFeatures.push_back(featExtr.extract(patches[patch]));
         
         patches.clear();
         
         //get cluster activations for the features
     //    dataActivation = codebook.getActivations(dataFeatures); 
     }
     dataFeaturesVec.push_back(dataFeatures);
   }
   cout << endl;
   convSVM.train(dataFeaturesVec, &dataset, codebook);
}


unsigned int CSVMClassifier::classifyConvSVM(Image* image){
      //cout << "nClasses = " << nClasses << endl;
   vector<Patch> patches;
   vector<Feature> dataFeatures;
   vector < vector<double> > dataActivation;
   
   if(settings.codebook == CB_CODEBOOK){
      
      vector<Patch> patches;
      vector<Feature> dataFeatures;
      
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

      for(size_t svmIdx = 0; svmIdx < dataset.getNumberClasses(); ++svmIdx)
         dataActivation.push_back(codebook.getQActivationsBackProp(dataFeatures, svmIdx)); 
   }else{      
//      dataActivation = deepCodebook->getActivations(image);
   }

   return convSVM.classify(dataActivation);
}


//train the KKT-SVM
void CSVMClassifier::trainClassicSVMs(){
   //cout << "Enteing classic svm training\n";
   unsigned int nTrainImages = dataset.getTrainSize();
   unsigned int nClasses = dataset.getNumberClasses(); 
   unsigned int nCentroids; 
   
   vector < vector<double> > datasetActivations;
   vector < Feature > dataFeatures;
   vector < Patch > patches;
   //cout << "datasetSize = " << datasetSize << endl;
   vector < vector<double> > dataKernel(nTrainImages);
   vector<double> dataActivation;

   //allocate space for more vectors
   datasetActivations.reserve(nTrainImages);
   //for all trainings imagages:
   for(size_t dataIdx = 0; dataIdx < nTrainImages; ++dataIdx){
      
      //extract patches
      if(settings.codebook == CB_CODEBOOK){
         patches = imageScanner.scanImage(dataset.getTrainImagePtr(dataIdx));
         dataActivation.clear();
         dataFeatures.clear();
         //clear previous features
         
         //allocate for new features
         dataFeatures.reserve(patches.size());
         
         //cout << patches[qIdx].size() << " patches" << endl;
         //extract features from all patches
         for(size_t patch = 0; patch < patches.size(); ++patch){
            dataFeatures.push_back(featExtr.extract(patches[patch]));
            //cout << "Patch at " << patches[patch].getX() << ", " << patches[patch].getY() << endl;
         }
         
         //cout << "Extracted " << patches.size() << "patches from the image\n";
         patches.clear();
         
         //get cluster activations for the features
         dataActivation = codebook.getActivations(dataFeatures); 
         dataFeatures.clear();
      
         patches.clear();
         

         //get cluster activations for the features
         datasetActivations.push_back(dataActivation);
         dataActivation.clear();
     }else{
        datasetActivations.push_back(deepCodebook->getActivations(dataset.getTrainImagePtr(dataIdx)));
     }
   }
   nClasses = dataset.getNumberClasses();
   nCentroids = datasetActivations[0].size();
   
   //cout << "nClasses = " << nClasses << endl;
   //cout << "nCentroids = " << nCentroids << endl;
   
   //cout << "Calculating similarities\n";
   //calculate similarity kernal between activation vectors
   for(size_t dIdx0 = 0; dIdx0 < nTrainImages; ++dIdx0){
      //cout << "done with similarity of " << dIdx0 << endl;
      for(size_t dIdx1 = 0; dIdx1 <= dIdx0; ++dIdx1){
         dataKernel[dIdx0].resize(dIdx0 + 1);
         double sum = 0;
	 
         if(settings.svmSettings.kernelType == RBF){
            


            for(size_t centr = 0; centr < nCentroids; ++centr){
               sum += (datasetActivations[dIdx0][centr] - datasetActivations[dIdx1][centr])*(datasetActivations[dIdx0][centr] - datasetActivations[dIdx1][centr]);
            }
            
            dataKernel[dIdx0][dIdx1] = exp((-1.0 * sum)/settings.svmSettings.sigmaClassicSimilarity);
            //dataKernel[dIdx1][dIdx0] = dataKernel[dIdx0][dIdx1];
            
         }else if (settings.svmSettings.kernelType == LINEAR){

            for(size_t cl = 0; cl < 1; ++cl){

               for(size_t centr = 0; centr < nCentroids; ++centr){
                  sum += (datasetActivations[dIdx0][centr] * datasetActivations[dIdx1][centr]);
               }
            }
            //cout << "Writing " << sum << " to " << dIdx0 << ", " << dIdx1 << endl;
            dataKernel[dIdx0][dIdx1] = sum;
            //dataKernel[dIdx1][dIdx0] = sum;
         }else
            cout << "CSVM::svm::Error! No valid kernel type selected! Try: RBF or LINEAR\n"  ;
         
      }
   }

   
   for(size_t cl = 0; cl < nClasses; ++cl){
      svms[cl].trainClassic(dataKernel, &dataset);  
   }
   classicTrainActivations = datasetActivations;
}


//classify an image using the KKT-SVM
unsigned int CSVMClassifier::classifyClassicSVMs(Image* image, bool printResults){
   unsigned int nClasses = dataset.getNumberClasses();
   //cout << "nClasses = " << nClasses << endl;
   vector<double> dataActivation;
   if(settings.codebook == CB_CODEBOOK){
      vector<Patch> patches;
      vector<Feature> dataFeatures;
      
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
      dataActivation = codebook.getActivations(dataFeatures); 
   }else{
      
      
      dataActivation = deepCodebook->getActivations(image);
   }
   //append centroid activations to activations from 0th quadrant.
   nClasses = dataset.getNumberClasses();   //normalize
   //cout << "Normalizing data" << endl;

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
   
   vector < Patch > patches;
   vector<double> dataActivation;
   //allocate space for more vectors
   datasetActivations.reserve(nTrainImages);
   //for all trainings imagages:
   for(size_t dataIdx = 0; dataIdx < nTrainImages; ++dataIdx){
      vector < Feature > dataFeatures;
      //extract patches
      if(settings.codebook == CB_CODEBOOK){
         patches = imageScanner.scanImage(dataset.getTrainImagePtr(dataIdx));
         dataActivation.clear();
         dataFeatures.clear();
         //clear previous features
         
         //allocate for new features
         dataFeatures.reserve(patches.size());
         
         //cout << patches[qIdx].size() << " patches" << endl;
         //extract features from all patches
         for(size_t patch = 0; patch < patches.size(); ++patch){
            dataFeatures.push_back(featExtr.extract(patches[patch]));
            //cout << "Patch at " << patches[patch].getX() << ", " << patches[patch].getY() << endl;
         }
         
         //cout << "Extracted " << patches.size() << "patches from the image\n";
         patches.clear();
         
         //get cluster activations for the features
         dataActivation = codebook.getActivations(dataFeatures); 
         dataFeatures.clear();
      
         patches.clear();
         

         //get cluster activations for the features
         datasetActivations.push_back(dataActivation);
         dataActivation.clear();
     }else{
        datasetActivations.push_back(deepCodebook->getActivations(dataset.getTrainImagePtr(dataIdx)));
     }
   }
   //cout << "Done getting activations\n";
   //train the Linear Netwok with the gained activations
   linNetwork.train(datasetActivations, &dataset);
}



unsigned int CSVMClassifier::lnClassify(Image* image){
   //cout << "nClasses = " << nClasses << endl;
   vector<Patch> patches;
   vector<Feature> dataFeatures;
   vector<double> dataActivation;
   //extract patches
  
   if(settings.codebook == CB_CODEBOOK){
      vector<Patch> patches;
      vector<Feature> dataFeatures;
      
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
      dataActivation = codebook.getActivations(dataFeatures); 
   }else{
      
      
      dataActivation = deepCodebook->getActivations(image);
   }
   

   return linNetwork.classify(dataActivation);

}
