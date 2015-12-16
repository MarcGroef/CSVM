
#include "experiment.h"


/*Optimze technique
 * use -pg compile flag
 * run as :
 * gprof -bp <program name > <output name>
 * 
 * 
 */

using namespace csvm;
using namespace std;

/*void showUsage(){
   cout << "CSVM: An experimental platform for the Convolutional Support Vector Machine architecture\n" <<
           "Usage: CSVM [settingsFile]\n" <<
           "Where:\n" <<
           "\tsettingsFile: location of settingsFile\n";
       
   
}*/

void help(){
   cout << "CSVM Python Functionality:\nvoid generateCodebook(char* settingsDir, char* codebook,char* dataDir)\ndouble run(char* settingsDir, char* codebook, char* dataDir)\n"  ;
}

void generateCodebook(char* settingsDir, char* codebook,char* dataDir){

   string codebookDir(codebook);
   CSVMClassifier c;
   c.setSettings(settingsDir);

   //load cifar10
   c.dataset.loadDataset(string(dataDir));
   
   c.constructCodebook();
   c.exportCodebook(codebookDir);
   cout << "Done constructing codebook\n";
}

double run(char* settingsDir, char* codebook, char* dataDir){
   srand(time(NULL));

   CSVMClassifier c;

   c.setSettings(settingsDir);
   
   c.dataset.loadDataset(string(dataDir));

   c.importCodebook(codebook);

   c.initSVMs();
   
   c.train();
   
   //cout << "Testing on trainingsset:\n";
   //Testing phase
   unsigned int nCorrect = 0;
   unsigned int nFalse = 0;

   unsigned int image;
   unsigned int trainSize = (unsigned int)c.dataset.getSize();
   
   unsigned int nImages = c.dataset.getTotalImages();
    
   for(size_t im = 0; im < 1000; ++im){
      
      image = trainSize + (rand() % (nImages-trainSize));
      
      unsigned int result = c.classify(c.dataset.getImagePtr(image));

      if((unsigned int)c.dataset.getImagePtr(image)->getLabelId() == result)
         ++nCorrect;
      else 
         ++nFalse;
      
   }
   
   return ((double)nFalse)/(nCorrect + nFalse);
}






