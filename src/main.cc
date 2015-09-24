
#include <csvm/csvm.h>
#include <iostream>
#include <time.h>

using namespace csvm;
using namespace std;

int main(int argc,char**argv){
   CSVMClassifier c;
   vector<string> imDirs;
   
   imDirs.push_back("../datasets/cifar-10-batches-bin/data_batch_1.bin");
   imDirs.push_back("../datasets/cifar-10-batches-bin/data_batch_2.bin");
   imDirs.push_back("../datasets/cifar-10-batches-bin/data_batch_3.bin");
   imDirs.push_back("../datasets/cifar-10-batches-bin/data_batch_4.bin");
   imDirs.push_back("../datasets/cifar-10-batches-bin/data_batch_5.bin");
   imDirs.push_back("../datasets/cifar-10-batches-bin/test_batch.bin");
   
   c.dataset.loadCifar10("../datasets/cifar-10-batches-bin/batches.meta.txt",imDirs);
   
   ImageScanner scanner;
   vector<Patch> newPatches;
   vector<Patch> patches;
   
   unsigned int nImages = (unsigned int) c.dataset.getSize();
   cout << nImages << " images loaded.\n";
   time_t time0 = clock();
   
   for(size_t idx = 0; idx < nImages; ++idx){
      newPatches = scanner.scanImage(c.dataset.getImagePtr(0),8,8,1,1);
      patches.insert(patches.end(),newPatches.begin(),newPatches.end());
   }
   
   cout << patches.size() << " patches collected! in " << (clock() - time0)/1000  << " ms\n";
   
   /*Image im;
   Image png;
   for(int i=0;i<20;i++){
      im = c.dataset.getImage(i);
      png = im.convertTo(CSVM_IMAGE_UCHAR_RGBA);
      png.exportImage(png.getLabel()+"cifarImages.png");
   }*/
   return 0;
}






