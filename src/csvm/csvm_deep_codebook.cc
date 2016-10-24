#include <csvm/csvm_deep_codebook.h>

using namespace std;
using namespace csvm;

// This class contains the deep codebook.
// calcSimilarity calculates activations from a feature to a centroid
// calculateSizes sets the layer sizes and the amount of layers. (The general architecture info)


//constructor
DeepCodebook::DeepCodebook(FeatureExtractor* fe, ImageScanner* imScanner, CSVMDataset* ds){
   
   scanner = imScanner;
   dataset = ds;
   featExtr = fe;

}

void DeepCodebook::setSettings(DCBSettings& s){
   this->settings = s;
   
   
   KMeans_settings mSets;
   mSets.nClusters = settings.nCentroids;
   
   mSets.alpha = 0;
   mSets.nIter = settings.nIter;
   
   kmeans.setSettings(mSets);

   
   calculateSizes(dataset->getImagePtr(0)->getWidth(), scanner->settings.patchWidth, scanner->settings.stride);
}


//*********** PRIVATE ********************

//Calculate activations of centroids given a feature
vector<float> DeepCodebook::calcSimilarity(Feature& p, vector<Centroid>& c){
   
   unsigned int nCentroids = c.size();
   unsigned int nDims = p.content.size();
   
   vector<float> activations(nCentroids, 0);
   vector<float> distances(nCentroids, 0);
   
   
   
   //Radial base function (Gaussian sheep)
	
   if(settings.simFunction == DCB_RBF){
      float dev;
      
      for(unsigned int word = 0; word < nCentroids; ++word){
         
         distances[word] = c[word].getDistanceSq(p);
         dev = exp(-1.0 * distances[word] / (settings.similaritySigma));
         activations[word] += dev;
         
      }
      
      //soft_assignment
   } else if (settings.simFunction == DCB_SOFT_ASSIGNMENT){
      //As done by Ng,Coates:
      
      float mean = 0.0;
      float xx = 0;
      for(size_t dim = 0; dim < nDims; ++dim){
         xx += p.content[dim] * p.content[dim];
      }
      
      for(unsigned int word = 0; word < nCentroids; ++word){
         float cc = 0.0;
         float xc = 0.0;
         
         for(size_t dim = 0; dim < nDims; ++dim){
            
            cc += c[word].content[dim] * c[word].content[dim];
            xc += c[word].content[dim] * p.content[dim];
            
         }
         //float dist = 0.0;
         //for(size_t dim = 0; dim < dataDims; ++dim){
         //   dist += (bow[cl][word].content[dim] - features[feat].content[dim]) *  (bow[cl][word].content[dim] - features[feat].content[dim]);
         //}
         
         distances[word] = sqrt(cc + (xx - (2 * xc))) ;
         //distances[word] = sqrt(dist);
         
         mean += distances[word];
      }
      mean /= (float)(nCentroids);
      for(unsigned int word = 0; word < nCentroids; ++word){
         activations[word] += ( mean - distances[word] > 0.0 ? mean - distances[word] : 0.0);
      }

   }else if(settings.simFunction == DCB_COSINE_SOFT_ASSIGNMENT){
		//With cosine distances
      
		
      float mean = 0.0;
      float xx = 0;
      for(size_t dim = 0; dim < nDims; ++dim){
         xx += p.content[dim] * p.content[dim];
      }
      
      for(unsigned int word = 0; word < nCentroids; ++word){
         float cc = 0.0;
         float xc = 0.0;
         
         for(size_t dim = 0; dim < nDims; ++dim){
            
            cc += c[word].content[dim] * c[word].content[dim];
            xc += c[word].content[dim] * p.content[dim];
            
         }
         
         //distances[word] = sqrt(cc + (xx - (2 * xc))) ;
         distances[word] = xc / (sqrt(xx*cc));
         //distances[word] = sqrt(dist);
         
         mean += distances[word];
      }
      mean /= (float)(nCentroids);
      for(unsigned int word = 0; word < nCentroids; ++word){
         activations[word] += ( mean - distances[word] > 0.0 ? mean - distances[word] : 0.0);
      }
	}
	//more iterations of max-clipping
   else if(settings.simFunction == DCB_SOFT_ASSIGNMENT_CLIPPING){
      float mean = 0.0;
      float xx = 0;
      
      for(size_t dim = 0; dim < nDims; ++dim){
         xx += p.content[dim] * p.content[dim];
      }
      for(unsigned int word = 0; word < nCentroids; ++word){
         float cc = 0.0;
         float xc = 0.0;
         
         for(size_t dim = 0; dim < nDims; ++dim){
            cc += c[word].content[dim] * c[word].content[dim];
            xc += c[word].content[dim] * p.content[dim];
            
         }
         
         
         distances[word] = sqrt(cc + (xx - (2 * xc))) ;
         //distances[word] = sqrt(dist);
         
         mean += distances[word];
      }
      mean /= (float)(nCentroids);
      for(unsigned int word = 0; word < nCentroids; ++word){
         activations[word] += ( mean - distances[word]> 0.0 ? mean - distances[word] : 0.0);
      }
      
      //and again!
      mean = 0;
      for(unsigned int word = 0; word < nCentroids; ++word){
         mean += activations[word];            
      }
      
      mean /= nCentroids;
      
      for(unsigned int word = 0; word < nCentroids; ++word){
         activations[word] = (activations[word] - mean) > 0.0 ? (activations[word] - mean) : 0.0;
      }
   }
   return activations;
}


//calculate the architecture sizes and number of layers
void DeepCodebook::calculateSizes(unsigned int imSize, unsigned int patchSize, unsigned int stride){
   
   unsigned int depth = 1;
   unsigned int fmSize = 1 + ((imSize - patchSize) / 2 );
   unsigned int plSize = fmSize / 2;
	//plSize = 2;    <-- if uncommented, it results in a regular BoW
   fmSizes.push_back(fmSize);
   plSizes.push_back(plSize);
   if(debugOut)
      cout << "fmSize0 = " << fmSize << endl;
   if(debugOut)
      cout << "plSize0 = " << plSize << endl;
   unsigned int tmpNCentroids = settings.nCentroids;
   nCentroids.push_back(tmpNCentroids);
   nRandomPatches.push_back(settings.nRandomPatches);
   
   for(size_t dIdx = 0; plSize > 2; ++dIdx, ++depth ){
      //tmpNCentroids /= 2;
      nCentroids.push_back(tmpNCentroids);
      fmSize = plSize;
      plSize = fmSize / 2;
      fmSizes.push_back(fmSize);
      plSizes.push_back(plSize);
      if(debugOut)
         cout << "fmSize = " << fmSize << endl;
      if(debugOut)
         cout << "plSize = " << plSize << endl;
      nRandomPatches.push_back(settings.nRandomPatches);
   }
   
   nLayers = depth;
   nCentroids = vector<unsigned int>(depth, settings.nCentroids);
   layerStack.resize(nLayers);
   //cout << "calculated settings for dcb\n" << "nLayers = " << nLayers << endl;;
}


//Next two function recursivly call each other throughout the layers. Maintaining a relative low memory consumption, while calculating all the maps

vector<float> DeepCodebook::calculatePoolMapAt(Image* im, unsigned int depth, unsigned int x, unsigned int y){   //vector elements are poolsum for each centroid
   //cout << "calc poolmap(" << depth << ") at " << x << ", " << y << endl; 
   unsigned int scanStride = fmSizes[depth] / plSizes[depth];
   unsigned int scanWidth = fmSizes[depth] % plSizes[depth] == 0 ? scanStride : scanStride + 1;
   
  
   vector<float> sum(nCentroids[depth], 0);
   for(size_t cvX = x * scanStride; cvX < (x + 1) * scanWidth; ++cvX){
      for(size_t cvY = y * scanStride; cvY < (y + 1) * scanWidth; ++cvY){
         
         vector<float> cvVals = calculateConvMapAt(im, depth, cvX, cvY);
         for(size_t centrIdx = 0; centrIdx < nCentroids[depth]; ++centrIdx){
            sum[centrIdx] += cvVals[centrIdx];
         }
      }
   }
   return sum;
}
//calc value at location of feature maps
vector<float> DeepCodebook::calculateConvMapAt(Image* im, unsigned int depth, unsigned int x, unsigned int y){  //feature map element at x,y for each centroid
   //cout << "calc cmap(" << depth << ") at " << x << ", " << y << endl; 
   if(depth == 0){//first layer, thus use image-patch extraction
      Feature f = featExtr->extract(scanner->getPatchAt(im, x, y));
      return calcSimilarity(f, layerStack[depth]);
   }else{//recursive step
      vector<float> pm = calculatePoolMapAt(im, depth - 1, x, y);
      Feature f(pm);
      return calcSimilarity(f, layerStack[depth]);
   }
}


//generate centroids for next layer. Collect features from random location at respective depth, and apply kmeans
void DeepCodebook::generateCentroids(){
   for(size_t depthIdx = 0; depthIdx < nLayers; ++depthIdx){
      vector<Feature> randomPatches;
      randomPatches.reserve(nRandomPatches[depthIdx]);
      srand(time(NULL));
      unsigned int totalImages = dataset->getTotalImages();
      
      if(depthIdx == 0){
         if(debugOut)
            cout << "Collecting patches..\n";
         for(size_t nIm = 0; nIm < nRandomPatches[depthIdx]; ++nIm){
            randomPatches.push_back(featExtr->extract(scanner->getRandomPatch(dataset->getImagePtr(rand() % totalImages))));
         }
         if(debugOut)
            cout << "Clustering at 0th layer..\n";
         layerStack[depthIdx] = kmeans.cluster(randomPatches, nCentroids[depthIdx]);
         
      }else{
         unsigned int scanWidth = plSizes[depthIdx - 1];
         if(debugOut)
            cout << "Collecting next-level patches..\n";
         for(size_t nIm = 0; nIm < nRandomPatches[depthIdx]; ++nIm){
            unsigned int imIdx = rand() % dataset->getTotalImages();
            vector<float> conv = calculatePoolMapAt(dataset->getImagePtr(imIdx), depthIdx - 1, rand() % scanWidth, rand() % scanWidth);
            
            randomPatches.push_back(Feature(conv));
         }
         layerStack[depthIdx] = kmeans.cluster(randomPatches, nCentroids[depthIdx]);
         
      }
   }
}


//normalize, be centering and stddev units
void standardize(vector<float>& vec){
   float mean = 0;
   float stddev = 0;
   
   unsigned int size = vec.size();
   for(size_t idx = 0; idx < size; ++idx){
      mean += vec[idx];
   }
   mean /= size;
   for(size_t idx = 0; idx < size; ++idx){
      stddev += (vec[idx] - mean) * (vec[idx] - mean);
   }
   stddev = sqrt(stddev + 0.001);
   stddev /= size;
   for(size_t idx = 0; idx < size; ++idx){
      vec[idx] = (vec[idx] - mean) / stddev;
   }
}

vector<float> DeepCodebook::getActivations(Image* im){
   vector<float> activations;
   unsigned int plSize = plSizes[nLayers - 1];
   for(size_t pmX = 0; pmX < plSize; ++pmX){
      for(size_t pmY = 0; pmY < plSize; ++pmY){
         vector<float> pmAct = calculatePoolMapAt(im, nLayers - 1, pmX, pmY);
         //standardize(pmAct);
         activations.insert(activations.begin(), pmAct.begin(), pmAct.end());
      }
   }
   return activations;
}



