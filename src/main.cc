
#include <csvm/csvm.h>
#include <iostream>
#include <time.h>

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
   //imDirs.push_back("../datasets/cifar-10-batches-bin/data_batch_2.bin");
   //imDirs.push_back("../datasets/cifar-10-batches-bin/data_batch_3.bin");
   //imDirs.push_back("../datasets/cifar-10-batches-bin/data_batch_4.bin");
   //imDirs.push_back("../datasets/cifar-10-batches-bin/data_batch_5.bin");
   //imDirs.push_back("../datasets/cifar-10-batches-bin/test_batch.bin");
   
   //load cifar10
   c.dataset.loadCifar10("../datasets/cifar-10-batches-bin/batches.meta.txt",imDirs);
   
  
   unsigned int nImages = (unsigned int) c.dataset.getSize();
   cout << nImages << " images loaded.\n";
   
   //measure cpu time
   time_t time0 = clock();
   
   for(size_t idx = 0; idx < nImages; ++idx){
      newPatches = scanner.scanImage(c.dataset.getImagePtr(idx),8,8,1,1);
      patches.insert(patches.end(),newPatches.begin(),newPatches.end());
   }
   cout << "patchwidth: " << patches[0].getWidth() << "\n";
   //print number of patches and time difference
   cout << patches.size() << " patches collected! in " << (clock() - time0)/1000  << " ms\n";
   vector<int> v = localBinPat.getLBP(patches[2], 0);
   for (int index =0 ; index < (int)(v.size());++index) {
	   cout << v[index] << "	";
   }
   
   
   return 0;
}






