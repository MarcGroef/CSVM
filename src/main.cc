#include <csvm/csvm.h>
#include <iostream>
#include <time.h>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <string> 
#include <sstream>


#include <csvm/csvm_interpolator.h>
/*Optimze technique
 * use -pg compile flag
 * run as :
 * gprof -bp <program name > <output name>
 * 
 * 
 */

using namespace csvm;
using namespace std;

void showUsage(){
   cout << "CSVM: An experimental platform for the Convolutional Support Vector Machine architecture\n" 
        << "Usage: CSVM [settingsFile]\n" 
        << "Where:\n" 
        << "\tsettingsFile: location of settingsFile\n";   
}


void printKernel(vector< vector<Feature> > kernels){
   size_t nKernels = 100;//= kernels.size();
   size_t nClusters = kernels[0].size();
   size_t nWords = kernels[0][0].content.size();
   
   for(size_t kIdx = 0; kIdx < nKernels; ++kIdx){
      cout << "******** Kernel " << kIdx << " **********\n";
      for(size_t cl = 0; cl < nClusters; ++cl){
         for(size_t w = 0; w < nWords; ++w)
            cout << kernels[kIdx][cl].content[w] << ", ";
         cout << endl;
      }
   }
}

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

int main(int argc,char**argv){
   
   //cout << "started main of CSVM" << endl;
   bool normalOut;
   if(argc!=2){
      showUsage();
      return 0;
   }
   
   srand(time(NULL));
   
   CSVMClassifier c;

   c.setSettings(argv[1]);
   normalOut = c.useOutput();
   
   c.dataset.loadDataset("../datasets/");

   
   /* BoW import export. Not needed for MLP
   if (c.getGenerateCB()){
      if (normalOut)
         cout << "constructing codebook" << endl;
      c.constructCodebook();
   } else {
      if (normalOut)
         cout << "importing codebook" << endl;
      c.importCodebook("LAST_USED.bin");
   }
   c.exportCodebook("LAST_USED.bin");
   */
   
  //This is not needed for the MLP training 
   /*c.constructCodebook();
   if(normalOut)
      cout << "initializing SVMs" << endl;
   c.initSVMs();
*/
   if(normalOut)
      cout << "training classifier" << endl;
   c.constructMLPs();
   c.train();

   //********************Testing phase on trainingset *****************************************
   unsigned int nCorrect = 0;
   unsigned int nFalse = 0;
   unsigned int nImages = c.dataset.getTrainSize();//(unsigned int) c.dataset.getSize();
   unsigned int nClasses = c.getNoClasses();
	
   vector<unsigned int> trainNumbers = c.dataset.getTrainImageNums();
   int validationSize = c.dataset.getTrainSize() * c.settings.mlpControlSettings.crossValidationSize;
   vector <vector <int> > classifiedAsTrain      ( nClasses +1, vector<int> ( nClasses +1, 0 ) );
   
   if(normalOut)
      cout << "Testing on trainingsset:\n";
      for(size_t im = 0;im < nImages-validationSize && im < 6000; ++im){
	  unsigned int result = c.classify(c.dataset.getTrainImagePtr(im));
	  unsigned int answer = c.dataset.getTrainImagePtr(im)->getLabelId();
	  //std::cout << "result: " << result << "answer: " << answer << std::endl;
	  if (result == answer){
	    ++nCorrect;
	  } else {
	    ++nFalse;
	  }

	  ++classifiedAsTrain[answer][result];
	  ++classifiedAsTrain[answer][nClasses];
	  ++classifiedAsTrain[nClasses][result];
      }
      if(normalOut)
	cout << nCorrect << " correct, and " << nFalse << " false classifications, out of " << nCorrect + nFalse << " images\n";
      if(normalOut)
	cout << "TrainSetScore: " << ((float)nCorrect * 100)/(nCorrect + nFalse) << "\% correct.\n";
      //****************************** Print ConfusionMatrix for TRAINSET *******************

      bool printConfusionMatrix = true;

      if (normalOut && printConfusionMatrix) {
	int   total;
	float precision;

	cout << "\n\n\t       Predicted:\t";
	for (size_t i=0; i<nClasses; i++){
	    if (c.dataset.getType() == DATASET_CIFAR10) cout << c.dataset.getLabel(i) << ((i<2) ? "\t" : "\t\t");   
	    else                                        cout << i << ((i<1) ? "\t" : "\t\t");   
	}
	cout << "Average:" << "\n\n    \tActual:\n";
	for (size_t i=0; i<nClasses; ++i){
	    total = 0;
	    if (c.dataset.getType() == DATASET_CIFAR10) cout << " \t" << c.dataset.getLabel(i) << ((i > 1) ? "\t" : "");
	    else                                        cout << " \t" << i << ((i > 1) ? "\t" : "\t");
	    for (size_t j=0; j<nClasses; ++j){
	      total += classifiedAsTrain[i][j];
	      cout << (((((j == 1 ) |( j == 2)) && i > 1)) ? "\t\t" : "\t\t") << fixed << classifiedAsTrain[i][j];
	    }
	    precision = (float)classifiedAsTrain[i][i] / total * 100;
	    cout << "\t\t" << precision << " %" << "\n\n\n";
	}
	cout << "\n\tPrecision:\t";
	for (size_t i=0; i<nClasses; ++i){
	    precision = (float) classifiedAsTrain[i][i] / classifiedAsTrain[nClasses][i] * 100;
	    cout << "\t" << fixed << precision << "";
	}
      }

    //***********************************Testing phase on validation set**********************************************************************************

   if(normalOut)
      cout << "\n\n\nOn validation set:\n\n";
   nCorrect = 0;
   nFalse = 0;
   
   vector <vector <int> > classifiedAsVal      ( nClasses +1, vector<int> ( nClasses +1, 0 ) );

   for(size_t im = nImages-validationSize; im < nImages; ++im){
      
      //cout << "classifying image " << image << endl;
      unsigned int result;
      unsigned int answer = c.dataset.getTrainImagePtr(im)->getLabelId();
      //cout << "\nAnswer: " << answer;
      result = c.classify(c.dataset.getTrainImagePtr(im));
      //if (result != answer) cout << "WRONG!    (answered " << result << ")\n\n\n";

      //cout << "result: " << result << endl;
      //cout << "classifying image \t" << image << ": " << c.dataset.getImagePtr(image)->getLabel() << " is classified as " << c.dataset.getLabel(result) << endl;

      
      if (result == answer){
         ++nCorrect;
      } else {
         ++nFalse;
      }

      ++classifiedAsVal[answer][result];
      ++classifiedAsVal[answer][nClasses];
      ++classifiedAsVal[nClasses][result];
      
   }

  
   
   //****************************** Print ConfusionMatrix for Validation *******************
   
   
   if(normalOut)cout << nCorrect << " correct, and " << nFalse << " false classifications, out of " << nCorrect + nFalse << " images\n";
   if(normalOut)cout << "Validationscore: " << ((float)nCorrect*100)/(nCorrect + nFalse) << "\% correct.\n";
   if(normalOut)cout << fixed << ((float)nCorrect)/(nCorrect + nFalse) << endl;
   

   if (normalOut && printConfusionMatrix) {
      int   total;
      float precision;

      cout << "\n\n\t       Predicted:\t";
      for (size_t i=0; i<nClasses; i++){
         if (c.dataset.getType() == DATASET_CIFAR10) cout << c.dataset.getLabel(i) << ((i<2) ? "\t" : "\t\t");   
         else                                        cout << i << ((i<1) ? "\t" : "\t\t");   
      }
      cout << "Average:" << "\n\n    \tActual:\n";
      for (size_t i=0; i<nClasses; ++i){
         total = 0;
         if (c.dataset.getType() == DATASET_CIFAR10) cout << " \t" << c.dataset.getLabel(i) << ((i > 1) ? "\t" : "");
         else                                        cout << " \t" << i << ((i > 1) ? "\t" : "\t");
         for (size_t j=0; j<nClasses; ++j){
            total += classifiedAsVal[i][j];
            cout << (((((j == 1 ) |( j == 2)) && i > 1)) ? "\t\t" : "\t\t") << fixed << classifiedAsVal[i][j];
         }
         precision = (float)classifiedAsVal[i][i] / total * 100;
         cout << "\t\t" << precision << " %" << "\n\n\n";
      }
      cout << "\n\tPrecision:\t";
      for (size_t i=0; i<nClasses; ++i){
         precision = (float) classifiedAsVal[i][i] / classifiedAsVal[nClasses][i] * 100;
         cout << "\t" << fixed << precision << "";
      }
   }
   
  
   //***********************************Testing phase on test set**********************************************************************************

   if(normalOut)
      cout << "\n\n\nOn test set:\n\n";
   nCorrect = 0;
   nFalse = 0;
   unsigned int testSize = (unsigned int)c.dataset.getTestSize();

   vector <vector <int> > classifiedAs      ( nClasses +1, vector<int> ( nClasses +1, 0 ) );
    /*
   time_t t = time(0); 
    struct tm * now = localtime( & t );
    string date = (now->tm_year + 1900) + '-' + (now->tm_mon + 1) + '-' + now->tm_mday;
   */
    time_t t = time(0); 
    struct tm * now = localtime( & t );
    string date = patch::to_string(now->tm_year + 1900) + '-' + patch::to_string(now->tm_mon + 1) + '-' + patch::to_string(now->tm_mday);
   
    for(size_t im = 0; im < testSize; ++im){
      
      //cout << "classifying image " << image << endl;
      unsigned int result;
      unsigned int answer = c.dataset.getTestImagePtr(im)->getLabelId();
      //cout << "\nAnswer: " << answer;
      result = c.classify(c.dataset.getTestImagePtr(im));
      if(c.settings.mlpControlSettings.saveWrongImages){
        if(answer != result){
            string name = date + "_" + patch::to_string(im) + "_"; 
            name += patch::to_string(answer);
            name += "->";
            name += patch::to_string(result);
            name += ".png";
            
            //cout << "this is the name: " << name << endl;
            c.dataset.getTestImagePtr(im)->exportImage(name);
        }
      }
        //if (result != answer) cout << "WRONG!    (answered " << result << ")\n\n\n";

      //cout << "result: " << result << endl;
      //cout << "classifying image \t" << image << ": " << c.dataset.getImagePtr(image)->getLabel() << " is classified as " << c.dataset.getLabel(result) << endl;

      
      if (result == answer){
         ++nCorrect;
      } else {
         ++nFalse;
      }

      ++classifiedAs[answer][result];
      ++classifiedAs[answer][nClasses];
      ++classifiedAs[nClasses][result];
      
   }

  
   
   //****************************** Print ConfusionMatrix for TESTSET *******************
   
   
   if(normalOut)cout << nCorrect << " correct, and " << nFalse << " false classifications, out of " << nCorrect + nFalse << " images\n";
   if(normalOut)cout << "TestSetScore: " << ((float)nCorrect*100)/(nCorrect + nFalse) << "\% correct.\n";
   if(normalOut)cout << fixed << ((float)nCorrect)/(nCorrect + nFalse) << endl;
   

   if (normalOut && printConfusionMatrix) {
      int   total;
      float precision;

      cout << "\n\n\t       Predicted:\t";
      for (size_t i=0; i<nClasses; i++){
         if (c.dataset.getType() == DATASET_CIFAR10) cout << c.dataset.getLabel(i) << ((i<2) ? "\t" : "\t\t");   
         else                                        cout << i << ((i<1) ? "\t" : "\t\t");   
      }
      cout << "Average:" << "\n\n    \tActual:\n";
      for (size_t i=0; i<nClasses; ++i){
         total = 0;
         if (c.dataset.getType() == DATASET_CIFAR10) cout << " \t" << c.dataset.getLabel(i) << ((i > 1) ? "\t" : "");
         else                                        cout << " \t" << i << ((i > 1) ? "\t" : "\t");
         for (size_t j=0; j<nClasses; ++j){
            total += classifiedAs[i][j];
            cout << (((((j == 1 ) |( j == 2)) && i > 1)) ? "\t\t" : "\t\t") << fixed << classifiedAs[i][j];
         }
         precision = (float)classifiedAs[i][i] / total * 100;
         cout << "\t\t" << precision << " %" << "\n\n\n";
      }
      cout << "\n\tPrecision:\t";
      for (size_t i=0; i<nClasses; ++i){
         precision = (float) classifiedAs[i][i] / classifiedAs[nClasses][i] * 100;
         cout << "\t" << fixed << precision << "";
      }
      cout << endl; 
   }
   return 0;
}