   
#include <csvm/csvm.h>
#include <iostream>
#include <time.h>
#include <cstdlib>

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

int main(int argc,char**argv){
   
   if(argc!=2){
      showUsage();
      return 0;
   }
   
   
   srand(time(NULL));
   
   CSVMClassifier c;

   c.setSettings(argv[1]);
   c.dataset.loadDataset("../datasets/");


   c.constructCodebook();
   
   //return 0;
   //c.importCodebook("goodmnist.bin");
   //c.exportCodebook("mnist1000.bin");
   //return 0;

   c.initSVMs();
   c.train();

   //********************Testing phase on trainingset *****************************************
   unsigned int nCorrect = 0;
   unsigned int nFalse = 0;
   unsigned int nImages = 50000;//(unsigned int) c.dataset.getSize();
   
   cout << "Testing on trainingsset:\n";
   for(size_t im = 0; im < 200 && im < nImages; ++im){
     
      unsigned int result;
      result = c.classify(c.dataset.getImagePtr(im));
      
      if((unsigned int)c.dataset.getImagePtr(im)->getLabelId() == result)
         ++nCorrect;
      else 
      ++nFalse;
          
   }
   cout << nCorrect << " correct, and " << nFalse << " false classifications, out of " << nCorrect + nFalse << " images\n";
   cout << "Score: " << ((double)nCorrect * 100)/(nCorrect + nFalse) << "\% correct.\n";
 
   
   //***********************************Testing phase on test set**********************************************************************************

   cout << "\n\nOn test set:\n\n";
   nCorrect = 0;
   nFalse = 0;
   unsigned int image;
   unsigned int trainSize = (unsigned int)c.dataset.getSize();
   unsigned int nClasses = c.getNoClasses();

   vector <vector <int> > classifiedAs      ( nClasses, vector<int> ( nClasses, 0 ) );

   for(size_t im = 0; im < 1000; ++im){
      image = trainSize + (rand() % (nImages - trainSize));
      
      //cout << "classifying image " << image << endl;
      unsigned int result;
      unsigned int answer = c.dataset.getImagePtr(image)->getLabelId();
      
      result = c.classify(c.dataset.getImagePtr(image));
      //cout << "result: " << result << endl;
      //cout << "classifying image \t" << image << ": " << c.dataset.getImagePtr(image)->getLabel() << " is classified as " << c.dataset.getLabel(result) << endl;

      
      if (result == answer){
         ++nCorrect;
      } else {
         ++nFalse;
      }
      ++classifiedAs[answer][result];
   }
   
   
   //****************************** Print ConfusionMatrix *******************
   
   
   cout << nCorrect << " correct, and " << nFalse << " false classifications, out of " << nCorrect + nFalse << " images\n";
   cout << "Score: " << ((double)nCorrect*100)/(nCorrect + nFalse) << "\% correct.\n";
   cout << fixed << ((double)nCorrect)/(nCorrect + nFalse) << endl;
   
   bool printConfusionMatrix = true;

   if (printConfusionMatrix) {
      int   total;
      double precision;

      cout << "\n\n\t       Predicted:\t";
      for (size_t i=0; i<nClasses; i++){
         cout << i /* c.dataset.getLabel(i)*/ << ((i<2) ? "\t" : "\t\t");   
      }
      cout << "Average:" << "\n\n    \tActual:\n";
      for (size_t i=0; i<nClasses; ++i){
         total = 0;
         cout << " \t" << i/*c.dataset.getLabel(i)*/ << ((i > 1) ? "\t" : "");
         for (size_t j=0; j<nClasses; ++j){
            total += classifiedAs[i][j];
            cout << (((((j == 1 ) |( j == 2)) && i > 1)) ? "\t\t" : "\t\t") << fixed << classifiedAs[i][j];// << "/" << total;
         }
         precision = (double)classifiedAs[i][i] / total * 100;
         cout << "\t\t" << precision << " %" << "\n\n\n";
      }
   }

   return 0;
}






