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
   mSets.liveROut = s.ROut;
   kmeans.setSettings(mSets);

   
   calculateSizes(dataset->getImagePtr(0)->getWidth(), scanner->settings.patchWidth, scanner->settings.stride);
}

void DeepCodebook::standardize(vector<double>& x, double sigmaFix){
   unsigned int size = x.size();
   double mean = 0;
   
   for(size_t idx = 0; idx != size; ++idx){
      mean += x[idx];
   }
   mean /= size;
   
   double sigma = 0;
   
   for(size_t idx = 0; idx != size; ++idx)
      sigma += (mean - x[idx]) * (mean - x[idx]);
   
   sigma /= size;
   sigma = sqrt(sigma + sigmaFix);
   
   for(size_t idx = 0; idx != size; ++idx){
      x[idx] = (x[idx] - mean) / sigma;
   }
   
}

vector<double> DeepCodebook::normalize(vector<double>& x){
  size_t size = x.size();
  double sum = 0;
  vector<double> normalized(size, 0);
  
  for(size_t idx = 0; idx != size; ++idx){
    sum += x[idx];
  }
  for(size_t idx = 0; idx != size && sum != 0; ++idx){
    normalized[idx] = x[idx] / sum;
  }
  return normalized;
}

//*********** PRIVATE ********************

//Calculate activations of centroids given a feature
vector<double> DeepCodebook::calcSimilarity(Feature& p, vector<Centroid>& c){
   
   unsigned int nCentroids = c.size();
   unsigned int nDims = p.content.size();
   
   vector<double> activations(nCentroids, 0);
   vector<double> distances(nCentroids, 0);
   //standardize(p.content,10);
   
   
   //Radial base function (Gaussian sheep)
	
   if(settings.simFunction == DCB_RBF){
      double dev;
      
      for(unsigned int word = 0; word < nCentroids; ++word){
         
         distances[word] = c[word].getDistanceSq(p);
         dev = exp(-1.0 * distances[word] / (settings.similaritySigma));
         activations[word] += dev;
         
      }
      
      //soft_assignment
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
         activations[word] += ( mean - distances[word] > 0.0 ? mean - distances[word] : 0.0);
      }

   }else if(settings.simFunction == DCB_COSINE_SOFT_ASSIGNMENT){
		//With cosine distances
      vector<double> normP = normalize(p.content);
      
		
      double mean = 0.0;
      double xx = 0;
      for(size_t dim = 0; dim < nDims; ++dim){
         xx += normP[dim] * normP[dim];
      }
      
      for(unsigned int word = 0; word < nCentroids; ++word){
         double cc = 0.0;
         double xc = 0.0;
         vector<double> normC = normalize(c[word].content);
         for(size_t dim = 0; dim < nDims; ++dim){
            
            cc += normC[dim] * normC[dim];
            xc += normC[dim] * normP[dim];
            
         }
         
         //distances[word] = sqrt(cc + (xx - (2 * xc))) ;
         distances[word] = xc / (sqrt(xx*cc));
         //distances[word] = sqrt(dist);
         
         mean += distances[word];
      }
      mean /= (double)(nCentroids);
      for(unsigned int word = 0; word < nCentroids; ++word){
         activations[word] += ( mean - distances[word] > 0.0 ? mean - distances[word] : 0.0);
      }
	}
	//more iterations of max-clipping
   else if(settings.simFunction == DCB_SOFT_ASSIGNMENT_CLIPPING){
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
         
         
         distances[word] = sqrt(cc + (xx - (2 * xc))) ;
         //distances[word] = sqrt(dist);
         
         mean += distances[word];
      }
      mean /= (double)(nCentroids);
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
   //standardize(activations, 0.01);
   return activations;
}


//calculate the architecture sizes and number of layers
void DeepCodebook::calculateSizes(unsigned int imSize, unsigned int patchSize, unsigned int stride){
   cout << "Imsize = " << imSize << endl;
   unsigned int depth = 1;
  // unsigned int fmSize = 1 + ((imSize - patchSize) / 2 );
   unsigned int fmSize = ((imSize - patchSize) / stride);
   unsigned int plSize = fmSize / 2;
	if(settings.architecture == DCB_ALPHA)
      plSize = 2;    // results in a regular BoW
      
   fmSizes.push_back(fmSize);
   plSizes.push_back(plSize);
   if(settings.debugOut)
      cout << "fmSize0 = " << fmSize << endl;
   if(settings.debugOut)
      cout << "plSize0 = " << plSize << endl;
   unsigned int tmpNCentroids = settings.nCentroids;
   nCentroids.push_back(tmpNCentroids);
   nRandomPatches.push_back(settings.nRandomPatches);
   
   for(size_t dIdx = 1; plSize > 2; ++dIdx, ++depth ){
      //tmpNCentroids /= 2;
      nCentroids.push_back(tmpNCentroids);
      fmSize = plSize;
      plSize = fmSize / 2;
      if((settings.architecture == DCB_BETA && dIdx == 1) || (settings.architecture == DCB_GAMMA && dIdx == 2))
         plSize = 2;

      fmSizes.push_back(fmSize);
      plSizes.push_back(plSize);
      if(settings.debugOut)
         cout << "fmSize" << dIdx << " = " << fmSize << endl;
      if(settings.debugOut)
         cout << "plSize" << dIdx << " = " << plSize << endl;
      nRandomPatches.push_back(settings.nRandomPatches);
   }
   
   nLayers = depth;
   nCentroids = vector<unsigned int>(depth, settings.nCentroids);
   layerStack.resize(nLayers);
   //cout << "calculated settings for dcb\n" << "nLayers = " << nLayers << endl;;
}


//Next two function recursivly call each other throughout the layers. Maintaining a relative low memory consumption, while calculating all the maps

vector<double> DeepCodebook::calculatePoolMapAt(Image* im, unsigned int depth, unsigned int x, unsigned int y){   //vector elements are poolsum for each centroid
   //cout << "calc poolmap(" << depth << ") at " << x << ", " << y << endl; 
   unsigned int scanStride = fmSizes[depth] / plSizes[depth];
   unsigned int scanWidth = fmSizes[depth] % plSizes[depth] == 0 ? scanStride : scanStride + 1;
   
   bool firstIsPassed = false;
   
   vector<double> sum(nCentroids[depth], 0);
   for(size_t cvX = x * scanStride; cvX < (x + 1) * scanWidth; ++cvX){
      for(size_t cvY = y * scanStride; cvY < (y + 1) * scanWidth; ++cvY){
        vector<double> cvVals = calculateConvMapAt(im, depth, cvX, cvY);
         
         if(settings.poolmethod == DCB_SUM){
            for(size_t centrIdx = 0; centrIdx < nCentroids[depth]; ++centrIdx){
                sum[centrIdx] += cvVals[centrIdx];
            }
         }else if(settings.poolmethod == DCB_MAX){
            for(size_t centrIdx = 0; centrIdx < nCentroids[depth]; ++centrIdx){
                if(!firstIsPassed || cvVals[centrIdx] > sum[centrIdx]){  //here 'sum' is used to store the max-values.
                  sum[centrIdx] = cvVals[centrIdx];
                }
                firstIsPassed = true;
            }
              
         }
         
      }
   }
   return sum;
}
//calc value at location of feature maps
vector<double> DeepCodebook::calculateConvMapAt(Image* im, unsigned int depth, unsigned int x, unsigned int y){  //feature map element at x,y for each centroid
   //cout << "calc cmap(" << depth << ") at " << x << ", " << y << endl; 
   if(depth == 0){//first layer, thus use image-patch extraction
      Feature f = featExtr->extract(scanner->getPatchAt(im, x, y));
      standardize(f.content,10);
      return calcSimilarity(f, layerStack[depth]);
   }else{//recursive step
      vector<double> pm = calculatePoolMapAt(im, depth - 1, x, y);
      Feature f(pm);
      standardize(f.content,10);
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
         if(settings.debugOut)
            cout << "Collecting patches..\n";
         for(size_t nIm = 0; nIm < nRandomPatches[depthIdx]; ++nIm){
            Feature f = featExtr->extract(scanner->getRandomPatch(dataset->getImagePtr(rand() % totalImages)));
            standardize(f.content,10);
            randomPatches.push_back(f);
         }
         if(settings.debugOut)
            cout << "Clustering at 0th layer..\n";
         cout << "Clustering " << randomPatches.size() << " patches with " << nCentroids[depthIdx] << " centroids" << endl;
         layerStack[depthIdx] = kmeans.cluster(randomPatches, nCentroids[depthIdx]);
         
      }else{
         unsigned int scanWidth = plSizes[depthIdx - 1];
         if(settings.debugOut)
            cout << "Collecting next-level patches..\n";
         for(size_t nIm = 0; nIm < nRandomPatches[depthIdx]; ++nIm){
            unsigned int imIdx = rand() % dataset->getTotalImages();
            vector<double> conv = calculatePoolMapAt(dataset->getImagePtr(imIdx), depthIdx - 1, rand() % scanWidth, rand() % scanWidth);
            Feature f(conv);
            standardize(f.content,10);
            randomPatches.push_back(f);
         }
         layerStack[depthIdx] = kmeans.cluster(randomPatches, nCentroids[depthIdx]);
         
      }
   }
}


vector<double> DeepCodebook::getActivations(Image* im){
   vector<double> activations;
   unsigned int plSize = plSizes[nLayers - 1];
   for(size_t pmX = 0; pmX < plSize; ++pmX){
      for(size_t pmY = 0; pmY < plSize; ++pmY){
         vector<double> pmAct = calculatePoolMapAt(im, nLayers - 1, pmX, pmY);
         //standardize(pmAct,0.01);
         activations.insert(activations.begin(), pmAct.begin(), pmAct.end());
      }
   }
   standardize(activations, 0.01);
   return activations;
}



