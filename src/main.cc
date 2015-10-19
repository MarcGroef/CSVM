
#include <csvm/csvm.h>
#include <iostream>
#include <time.h>


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
   cout << "CSVM: An experimental platform for the Convolutional Support Vector Machine architecture\n" <<
           "Usage: CSVM [settingsFile]\n" <<
           "Where:\n" <<
           "\tsettingsFile: location of settingsFile\n";
       
   
}

int main(int argc,char**argv){
   
   if(argc!=2){
      showUsage();
      return 0;
   }
   
   CSVMClassifier c;
   ImageScanner scanner;
   vector<Patch> newPatches;
   vector<Patch> patches;
   LBPDescriptor localBinPat;

   //load settingsFile
   c.setSettings(argv[1]);
   
   
   
   //setup cifar10 data directories
   vector<string> imDirs;
   
   imDirs.push_back("../datasets/cifar-10-batches-bin/data_batch_1.bin");
   imDirs.push_back("../datasets/cifar-10-batches-bin/data_batch_2.bin");
   imDirs.push_back("../datasets/cifar-10-batches-bin/data_batch_3.bin");
   imDirs.push_back("../datasets/cifar-10-batches-bin/data_batch_4.bin");
   imDirs.push_back("../datasets/cifar-10-batches-bin/data_batch_5.bin");
   imDirs.push_back("../datasets/cifar-10-batches-bin/test_batch.bin");
   
   //load cifar10
   c.dataset.loadCifar10("../datasets/cifar-10-batches-bin/batches.meta.txt",imDirs);
   
  
   unsigned int nImages = (unsigned int) c.dataset.getSize();
   cout << nImages << " images loaded.\n";
   
   //measure cpu time
   cout << "Start timing\n";
   time_t time0 = clock();
   
   c.constructCodebook();
   c.exportCodebook("codebook.bin");

   //c.importCodebook("codebook.bin");

   //svm stuff
   c.initSVMs();
   //c.trainSVMs();
   
   vector< vector< Feature> > trainActivations = c.trainClassicSVMs(2.0);
   unsigned int nCorrect = 0;
   unsigned int nFalse = 0;
   for(size_t im = 0; im < 1000; ++im){
      //unsigned int result = c.classify(c.dataset.getImagePtr(im));
      unsigned int result = c.classifyClassicSVMs(c.dataset.getImagePtr(im), trainActivations);
      cout << "classifying image " << im << ": " << c.dataset.getImagePtr(im)->getLabelId() << " is classified as " << result << endl;
      if((unsigned int)c.dataset.getImagePtr(im)->getLabelId() == im)
         ++nCorrect;
      else 
         ++nFalse;
   }
   cout << nCorrect << " correct, and " << nFalse << " false classifications, out of " << 1000 << "images\n";
   cout << "Processed in " << (double)(clock() - time0)/1000  << " ms\n";

   return 0;
}






