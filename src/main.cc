   
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
   cout << "CSVM: An experimental platform for the Convolutional Support Vector Machine architecture\n" <<
           "Usage: CSVM [settingsFile]\n" <<
           "Where:\n" <<
           "\tsettingsFile: location of settingsFile\n";
       
   
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
   //cout << "Starting program..\n";
  /* if(argc!=2){
      showUsage();
      return 0;
//    }*/
   srand(time(NULL));
   string dataDir = "../datasets/";//argv[2];
   CSVMClassifier c;
   ImageScanner scanner;
   vector<Patch> newPatches;
   vector<Patch> patches;
   LBPDescriptor localBinPat;

   //load settingsFile
   c.setSettings(argv[1]);
   
   
   
   //setup cifar10 data directories
   vector<string> imDirs;
   
   imDirs.push_back(dataDir + "cifar-10-batches-bin/data_batch_1.bin");
   imDirs.push_back(dataDir + "cifar-10-batches-bin/data_batch_2.bin");
   imDirs.push_back(dataDir + "cifar-10-batches-bin/data_batch_3.bin");
   imDirs.push_back(dataDir + "cifar-10-batches-bin/data_batch_4.bin");
   imDirs.push_back(dataDir + "cifar-10-batches-bin/data_batch_5.bin");
   imDirs.push_back(dataDir + "cifar-10-batches-bin/test_batch.bin");
   
   //load cifar10
   //c.dataset.loadCifar10(dataDir + "cifar-10-batches-bin/batches.meta.txt",imDirs);
   c.dataset.loadMNIST(dataDir + "mnist/");
   //cout << "ready to work!\n";
   
   unsigned int nImages = 50000;//(unsigned int) c.dataset.getSize();
   //cout << nImages << " images loaded.\n";
   
   
   
   
   //measure cpu time
   //cout << "Start timing\n";
   time_t time0 = clock();
   
   //c.constructCodebook();
   //cout << "Constructed codebooks in " << (double)(clock() - time0)/1000  << " ms\n";
  
   //c.exportCodebook("codebook50000HOG.bin");
   //return 0;
   //cout << "Constructed Codebook!\n";
   //return 0;
   c.importCodebook("codebook50000HOG.bin");

   c.initSVMs();
   //cout << "Start training SVMs\n";
   //train convolutional SVMs
   //c.trainSVMs();
   
   //train classic SVM
   vector< vector< vector<double> > > trainActivations;
   if(c.useClassicSVM()){
      cout << "Training classic SVM\n";
      trainActivations = c.trainClassicSVMs();
      
   }else{
      
      cout << "Training Conv SVM\n";
      c.trainSVMs();
   }
   //printKernel(trainActivations);
   cout << "Testing on trainingsset:\n";
   //Testing phase
   unsigned int nCorrect = 0;
   unsigned int nFalse = 0;

   for(size_t im = 0; im < 200 && im < nImages; ++im){
      //classify using convolutional SVMs 
      //unsigned int result = c.classify(c.dataset.getImagePtr(im));
      //classify using classic SVMs
      unsigned int result;
      if(c.useClassicSVM())
         result = c.classifyClassicSVMs(c.dataset.getImagePtr(im), trainActivations, false );
      else
         result = c.classify(c.dataset.getImagePtr(im));
      //cout << "classifying image \t" << im << ": " << c.dataset.getImagePtr(im)->getLabel() << " is classified as " << c.dataset.getLabel(result) << endl;

      if((unsigned int)c.dataset.getImagePtr(im)->getLabelId() == result)
         ++nCorrect;
      else 
      ++nFalse;
   //       
   }
   cout << nCorrect << " correct, and " << nFalse << " false classifications, out of " << nCorrect + nFalse << " images\n";
   cout << "Score: " << ((double)nCorrect * 100)/(nCorrect + nFalse) << "\% correct.\n";
   
   
   //*********************************************************************************************************************
   
   
   //cout << "Testing on Testset:\n";
   //Testing phase
   cout << "On test set:\n";
   nCorrect = 0;
   nFalse = 0;
   unsigned int image;
   unsigned int trainSize = (unsigned int)c.dataset.getSize();
   for(size_t im = 0; im < 500; ++im){
      image = trainSize + (rand() % (nImages - trainSize));
      //cout << "Testing image " << image << ".. ";
      //classify using convolutional SVMs
      //unsigned int result = c.classify(c.dataset.getImagePtr(im));
      //classify using classic SVMs
      
      unsigned int result;
      if(c.useClassicSVM())
         result = c.classifyClassicSVMs(c.dataset.getImagePtr(image), trainActivations, false /*im > 50200 - 0 - 10*/);
      else
         result = c.classify(c.dataset.getImagePtr(image));
      //cout << "classifying image \t" << image << ": " << c.dataset.getImagePtr(image)->getLabel() << " is classified as " << c.dataset.getLabel(result) << endl;
      if((unsigned int)c.dataset.getImagePtr(image)->getLabelId() == result){
         ++nCorrect;
         //cout << "Correct!\n";
      }
      else {
         ++nFalse;
         //cout << "False!\n";
      }
   }
   cout << nCorrect << " correct, and " << nFalse << " false classifications, out of " << nCorrect + nFalse << " images\n";
   cout << "Score: " << ((double)nCorrect*100)/(nCorrect + nFalse) << "\% correct.\n";
   cout << fixed << ((double)nCorrect)/(nCorrect + nFalse) << endl;
   
   
   //cout << "Processed in " << (double)(clock() - time0)/1000  << " ms\n";

   return 0;
}






