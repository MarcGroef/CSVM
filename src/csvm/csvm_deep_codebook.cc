#include <csvm/csvm_deep_codebook.h>

using namespace std;
using namespace csvm;

DeepCodebook::DeepCodebook(ImageScanner* imScanner, CSVMDataset* ds,unsigned int imSize, unsigned int patchSize, unsigned int stride){

   KMeans_settings mSets;
   mSets.nClusters = 0;
   
   mSets.alpha = 0;
   mSets.nIter = 15;
   kmeans.setSettings(mSets);
   
   calculateSizes(imSize, patchSize, stride);
   scanner = imScanner;
   dataset = ds;
   
   
}


//*********** PRIVATE ********************

void DeepCodebook::calculateSizes(unsigned int imSize, unsigned int patchSize, unsigned int stride){
   unsigned int depth = 1;
   unsigned int fmSize = 1 + ((imSize - patchSize) / 2 );
   unsigned int plSize = fmSize / 2;
   fmSizes.push_back(fmSize);
   plSizes.push_back(plSize);
   
   for(size_t dIdx = 0; plSize < 2; ++dIdx, ++depth ){
      fmSize = plSize;
      plSize = fmSize / 2;
      
      fmSizes.push_back(fmSize);
      plSizes.push_back(plSize);
      
   }
   nLayers = depth;
   nCentroids = vector<unsigned int>(depth, 100);
}

vector<double> calculatePoolMapAt(unsigned int imIdx, unsigned int depth, unsigned int x, unsigned int y){   //vector elements are poolsum for each centroid
   unsigned int scanStride = fmSizes[depth] / plSizes[depth];
   unsigned int scanWidth = fmSizes[depth] % plSizes == 0 ? scanStride : scanStride + 1;
   
   if(depth == 0){  //first layer, thus use image-patch extraction
      
   }else{  //recursive step
      
   }
}

vector<double> calculateConvMapAt(unsigned int imIdx, unsigned int depth, unsigned int x, unsigned int y){  //feature map element at x,y for each centroid
   if(depth == 0){//first layer, thus use image-patch extraction
      
   }else{//recursive step
      
   }
}


