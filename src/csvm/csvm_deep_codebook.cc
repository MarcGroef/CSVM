#include <csvm/csvm_deep_codebook.h>

using namespace std;
using namespace csvm;

DeepCodebook::DeepCodebook(FeatureExtractor* fe, ImageScanner* imScanner, CSVMDataset* ds,unsigned int imSize, unsigned int patchSize, unsigned int stride){

   KMeans_settings mSets;
   mSets.nClusters = 0;
   
   mSets.alpha = 0;
   mSets.nIter = 15;
   kmeans.setSettings(mSets);
   
   calculateSizes(imSize, patchSize, stride);
   scanner = imScanner;
   dataset = ds;
   featExtr = fe;
   
}


//*********** PRIVATE ********************
vector<double> DeepCodebook::calcSimilarity(Feature& p, vector<Centroid>& c){
   
   unsigned int nCentroids = c.size();
   unsigned int nDims = p.content.size();
   
   vector<double> activations(nCentroids, 0);
   vector<double> distances(nCentroids, 0);
   
   
   
   
   if(settings.simFunction == DCB_RBF){
      double dev;
      
      for(unsigned int word = 0; word < nCentroids; ++word){
         
         distances[word] = c[word].getDistanceSq(p);
         dev = exp(-1.0 * distances[word] / (settings.similaritySigma));
         activations[word] += dev;
         
      }
      
   } else if (settings.simFunction == DCB_SOFT_ASSIGNMENT){
      //As done by Ng,Coates:
      
      double mean = 0.0;
      double xx = 0;
      for(size_t dim = 0; dim < nDims; ++dim){
         xx += p.content[dim] * p.content[dim];
      }
      
      for(unsigned int word = 0; word < nCentroids; ++word){
         double cc = 0.0;
         double xc = 0.0;
         
         for(size_t dim = 0; dim < nDims; ++dim){
            
            cc += c[word].content[dim] * c[word].content[dim];
            xc += c[word].content[dim] * p.content[dim];
            
         }
         //double dist = 0.0;
         //for(size_t dim = 0; dim < dataDims; ++dim){
         //   dist += (bow[cl][word].content[dim] - features[feat].content[dim]) *  (bow[cl][word].content[dim] - features[feat].content[dim]);
         //}
         
         distances[word] = sqrt(cc + (xx - (2 * xc))) ;
         //distances[word] = sqrt(dist);
         
         mean += distances[word];
      }
      mean /= (double)(nCentroids);
      for(unsigned int word = 0; word < nCentroids; ++word){
         activations[word] += ( mean - distances[word]> 0.0 ? mean - distances[word] : 0.0);
      }

   }
   return activations;
}

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

vector<double> DeepCodebook::calculatePoolMapAt(unsigned int imIdx, unsigned int depth, unsigned int x, unsigned int y){   //vector elements are poolsum for each centroid
   unsigned int scanStride = fmSizes[depth] / plSizes[depth];
   unsigned int scanWidth = fmSizes[depth] % plSizes[depth] == 0 ? scanStride : scanStride + 1;
   
   if(depth == 0){  //first layer, thus use image-patch extraction
      double sum = 0;
      
   }else{  //recursive step
      
   }
}

vector<double> DeepCodebook::calculateConvMapAt(unsigned int imIdx, unsigned int depth, unsigned int x, unsigned int y){  //feature map element at x,y for each centroid
   if(depth == 0){//first layer, thus use image-patch extraction
      Feature f = featExtr->extract(scanner->getPatchAt(dataset->getImagePtr(imIdx), x, y));
      return calcSimilarity(f, layerStack[depth]);
   }else{//recursive step
      
   }
}


