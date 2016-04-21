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
   
   settings.codebookSettings.kmeansSettings.normalOut = normalOut;
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
   
   mlp.setSettings(settings.mlpSettings);
}

//Train the system
void CSVMClassifier::train(){
   switch(settings.classifier){
      case CL_SVM:
         if(normalOut) cout << "Training SVM..\n";
         trainClassicSVMs();
         
         break;
      case CL_CSVM:
         if(normalOut) cout << "Training Conv SVM..\n";
         trainConvSVMs();
         
         break;
      case CL_LINNET:
         if(normalOut) cout << "Training LinNet..\n";
         trainLinearNetwork();
		case CL_MLP:
			if(normalOut) cout << "Training MLP..\n";
			trainMLP();
         break;
      default:
         cout << "WARNING! couldnt recognize selected classifier!\n";
   }
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
		case CL_MLP:
			result = mlpClassify(im);
			break;
   }
   return result;
}


//export the current codebook to file (Only works for the normal codebook, not yet for the deep bow)
void CSVMClassifier::exportCodebook(string filename){
   codebook.exportCodebook(filename);
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


void CSVMClassifier::trainMLP(){
    unsigned int nPatches = settings.scannerSettings.nRandomPatches;
    unsigned int imageHeight = settings.datasetSettings.imHeight;
    unsigned int imageWidth = settings.datasetSettings.imWidth;
    unsigned int patchHeight = settings.scannerSettings.patchHeight;
    unsigned int patchWidth = settings.scannerSettings.patchWidth;
    unsigned int stride = settings.scannerSettings.stride;
    
	vector<Feature> pretrainDump;
	vector<Feature> testData;
  
	vector<int> classes = vector<int>(10,1);

	//---------------start validation set--------------------
	int crossvalidationSize = dataset.getTrainSize() * settings.mlpSettings.crossValidationSize;
	int noPatchesPerImage = ((int)((imageHeight - patchHeight) / stride) + 1) * ((int)((imageWidth - patchWidth) / stride) + 1) ;

     vector<Patch> patches;
	 vector<Feature> validationSet;    
    
     validationSet.reserve(noPatchesPerImage*crossvalidationSize);
  
	for(int i = dataset.getTrainSize() - crossvalidationSize; i < dataset.getTrainSize();i++){
		Image* im = dataset.getTrainImagePtr(i);
		 
		//extract patches
		patches = imageScanner.scanImage(im);
      
		//extract features from all patches
		for(size_t patch = 0; patch < patches.size(); ++patch)
			validationSet.push_back(featExtr.extract(patches[patch]));
	}
	
   std::cout << "Feature extraction training set..." << std::endl;
   for(size_t pIdx = 0; pIdx < nPatches; ++pIdx){
	  //std::cout << pIdx << std::endl;
      //patches = imageScanner.getRandomPatches(dataset.getImagePtrFromClass(im, cl));
      Patch patch = imageScanner.getRandomPatch(dataset.getTrainImagePtr(rand() %  (int)(dataset.getTrainSize()*1-settings.mlpSettings.crossValidationSize)));
      Feature newFeat = featExtr.extract(patch);
      pretrainDump.push_back(newFeat);//insert(pretrainDump[cl].end(),features.begin(),features.end());      
   }
   mlp.train(pretrainDump,validationSet,noPatchesPerImage);
}

unsigned int CSVMClassifier::mlpClassify(Image* im){
	
	  vector<Patch> patches;
      vector<Feature> dataFeatures;
      
      //extract patches
      patches = imageScanner.scanImage(im);

      //allocate for new features
      dataFeatures.reserve(patches.size());
      
      //extract features from all patches
      for(size_t patch = 0; patch < patches.size(); ++patch)
         dataFeatures.push_back(featExtr.extract(patches[patch]));
		
		return mlp.classify(dataFeatures);
}

//construct a codebook using the current dataset
void CSVMClassifier::constructCodebook(){
   cout << "hoi\n";
   if(settings.codebook == CB_DEEPCODEBOOK){
      constructDeepCodebook();
      return;
   }else if(settings.codebook == CB_MLP){
      //cout << "should be training the mlp...\n";
      //trainMLP();
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

   if(debugOut) cout << "Collected " << pretrainDump.size()<< " features\n";
   codebook.constructCodebook(pretrainDump);
   
   if(debugOut) cout << "done constructing codebook using "  << settings.scannerSettings.nRandomPatches << " patches\n";
   
   pretrainDump.clear();
}


void CSVMClassifier::trainConvSVMs(){
   unsigned int nTrainImages = dataset.getTrainSize();
   vector < vector < double > > datasetActivations;
   vector < Feature > dataFeatures;
   vector < Patch > patches;
   
   //allocate space for more vectors
   datasetActivations.reserve(nTrainImages);
   //for all trainings imagages:
   for(size_t dataIdx = 0; dataIdx < nTrainImages; ++dataIdx){
      vector<double> dataActivation;
      //extract patches
      
      if(settings.codebook == CB_CODEBOOK){
         
         patches = imageScanner.scanImage(dataset.getTrainImagePtr(dataIdx));
         dataActivation.clear();
         dataFeatures.clear();
         //clear previous features
         
         //allocate for new features
         dataFeatures.reserve(patches.size());
         
         //extract features from all patches
         for(size_t patch = 0; patch < patches.size(); ++patch){
            dataFeatures.push_back(featExtr.extract(patches[patch]));
         }
         
         patches.clear();
         
         //get cluster activations for the features
         dataActivation = codebook.getActivations(dataFeatures); 
         dataFeatures.clear();
      
         patches.clear();
         

         //get cluster activations for the features
         datasetActivations.push_back(dataActivation);
         dataActivation.clear();
     }else if(settings.codebook == CB_DEEPCODEBOOK){
        datasetActivations.push_back(deepCodebook->getActivations(dataset.getTrainImagePtr(dataIdx)));
     }else if(settings.codebook == CB_MLP){
        /*
        patches = imageScanner.scanImage(dataset.getTrainImagePtr(dataIdx));
         dataActivation.clear();
         dataFeatures.clear();
         //clear previous features
         
         //allocate for new features
         dataFeatures.reserve(patches.size());
         
         //extract features from all patches
         for(size_t patch = 0; patch < patches.size(); ++patch){
            dataFeatures.push_back(featExtr.extract(patches[patch]));
         */}
         
         patches.clear();
          
         dataFeatures.clear();
      
         patches.clear();
         

         //get cluster activations for the features
         datasetActivations.push_back(dataActivation);
         dataActivation.clear();
   }
   convSVM.train(datasetActivations, &dataset);
}


unsigned int CSVMClassifier::classifyConvSVM(Image* image){
   vector<Patch> patches;
   vector<Feature> dataFeatures;
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
   }else if(settings.codebook == CB_DEEPCODEBOOK){      
      dataActivation = deepCodebook->getActivations(image);
   }else if(settings.codebook == CB_MLP){
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
      cout << "class: train mlp\n";  
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
         patches = imageScanner.scanImage(dataset.getTrainImagePtr(dataIdx));
         dataActivation.clear();
         dataFeatures.clear();
         //clear previous features
         
         //allocate for new features
         dataFeatures.reserve(patches.size());
         
         //extract features from all patches
         for(size_t patch = 0; patch < patches.size(); ++patch){
            dataFeatures.push_back(featExtr.extract(patches[patch]));
            //cout << "Patch at " << patches[patch].getX() << ", " << patches[patch].getY() << endl;
         }
         
         
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
   
   if(debugOut) cout << "nClasses = " << nClasses << endl;
   if(debugOut) cout << "nCentroids = " << nCentroids << endl;
   
   if(debugOut) cout << "Calculating similarities\n";
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


//classify an image using the regular-SVM
unsigned int CSVMClassifier::classifyClassicSVMs(Image* image, bool printResults){
   unsigned int nClasses = dataset.getNumberClasses();
   
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
         }
         
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
   //train the Linear Netwok with the gained activations
   linNetwork.train(datasetActivations, &dataset);
}



unsigned int CSVMClassifier::lnClassify(Image* image){
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
