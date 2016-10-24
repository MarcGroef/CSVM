#include <csvm/csvm_codebook.h>

using namespace std;
using namespace csvm;

/* This class implements the Bag-of-Visual-Words model (a.k.a codebook)
 * It contains the centroids and can map a feature to an activation vector, given the centroids
 * 
 * 
 * 
 */

//unions to make life easier while reading/writing binary files

void Codebook::standardize(vector<float>& x, float sigmaFix){
   unsigned int size = x.size();
   float mean = 0;
   
   for(size_t idx = 0; idx != size; ++idx){
      mean += x[idx];
   }
   mean /= size;
   
   float sigma = 0;
   
   for(size_t idx = 0; idx != size; ++idx)
      sigma += (mean - x[idx]) * (mean - x[idx]);
   
   sigma /= size;
   sigma = sqrt(sigma + sigmaFix);
   
   for(size_t idx = 0; idx != size; ++idx){
      x[idx] = (x[idx] - mean) / sigma;
   }
   
}


union charInt{
   char chars[4];
   unsigned int intVal;
};

union charfloat{
   char chars[8];
   float floatVal;
};

Codebook::Codebook(){
   nClasses = 10;
}

void Codebook::setSettings(Codebook_settings s){
   settings = s;
   kmeans.setSettings(s.kmeansSettings);
   akmeans.setSettings(s.akmeansSettings);
}

bool Codebook::getGenerate(){
   return settings.generate;
}

Centroid Codebook::getCentroid(int centrIdx){
   return bow[centrIdx];
}

void Codebook::constructCodebook(vector<Feature> featureset){
   //cout << "constructing codebook for label " << labelId << " in ";
   unsigned int nFeatures = featureset.size();
   
   if(settings.whitening)
      w.analyze(featureset);
   
   for(size_t fIdx = 0; fIdx != nFeatures; ++fIdx){
      if(settings.standardize)
         standardize(featureset[fIdx].content, 10);
      if(settings.whitening)
         w.transform(featureset[fIdx]);
   }
   switch(settings.method){
      case LVQ_Clustering:
         //bow[labelId] = lvq.cluster(featureset, labelId, settings.numberVisualWords, 0.1,120);
         break;
      case KMeans_Clustering:
         bow = kmeans.cluster(featureset, settings.numberVisualWords);
         break;
      case AKMeans_Clustering:
         bow = akmeans.cluster(featureset, settings.numberVisualWords, nClasses);
         break;
   }
   
}




unsigned int Codebook::getNClasses(){
      return nClasses;
}


unsigned int Codebook::getNCentroids(){
   return settings.numberVisualWords;
}

vector<vector <float> > Codebook::getCentroidByClassContributions() {
	return akmeans.getClusterClassContributions();
}

vector<float> Codebook::getCentroidByClassContributions(int cl) {
	return akmeans.getClusterClassContributions(cl);
}

vector<float> Codebook::getCentroidByClassContributions(Feature feat) {
	return akmeans.getClusterClassContributions(feat);
}

vector< float > Codebook::getActivations(vector<Feature> features){
  return getQActivations(features);
   //if(features.size() % 4 != 0)
   //   cout << "Warning: patches do not perfectly fit into quadrants..\n";

   vector< float> activations(settings.numberVisualWords, 0.0);
   unsigned int dataDims = features[0].content.size();
   vector<float> distances(settings.numberVisualWords);
   float dev;
   unsigned int nFeatures = features.size();
   float mean;
   
   float xx = 0.0;
   float cc;
   float xc;
   
   for(size_t feat = 0; feat < nFeatures; ++feat){
      
      
      if (settings.simFunction == SOFT_ASSIGNMENT){
         xx = 0;
         for(size_t dim = 0; dim < dataDims; ++dim){
            xx += features[feat].content[dim] * features[feat].content[dim];
         }
      }
      
      mean = 0.0;
      //cout << "cl" << cl << ": ";
      
      if(settings.simFunction == CB_RBF){
         
         for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
            
            distances[word] = bow[word].getDistanceSq(features[feat]);
            dev = exp(-1.0 * distances[word] / (settings.similaritySigma));
            activations[word] += dev;
            
         }
         
      } else if (settings.simFunction == SOFT_ASSIGNMENT){
         //As done by Ng,Coates:
         
         mean = 0.0;
         for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
            cc = 0.0;
            xc = 0.0;
            
            for(size_t dim = 0; dim < dataDims; ++dim){
               cc += bow[word].content[dim] * bow[word].content[dim];
               xc += bow[word].content[dim] * features[feat].content[dim];
               
            }
            //float dist = 0.0;
            //for(size_t dim = 0; dim < dataDims; ++dim){
            //   dist += (bow[cl][word].content[dim] - features[feat].content[dim]) *  (bow[cl][word].content[dim] - features[feat].content[dim]);
            //}
            
            distances[word] = sqrt(cc + (xx - (2 * xc))) ;
            //distances[word] = sqrt(dist);
            
            mean += distances[word];
         }
         mean /= (float)(settings.numberVisualWords);
         for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
            activations[word] += ( mean - distances[word]> 0.0 ? mean - distances[word] : 0.0);
         }
		}else if(settings.simFunction == COSINE_SOFT_ASSIGNMENT){
         
         mean = 0.0;
         for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
            cc = 0.0;
            xc = 0.0;
            
            for(size_t dim = 0; dim < dataDims; ++dim){
               cc += bow[word].content[dim] * bow[word].content[dim];
               xc += bow[word].content[dim] * features[feat].content[dim];
               
            }
            //float dist = 0.0;
            //for(size_t dim = 0; dim < dataDims; ++dim){
            //   dist += (bow[cl][word].content[dim] - features[feat].content[dim]) *  (bow[cl][word].content[dim] - features[feat].content[dim]);
            //}
            distances[word] = xc / (sqrt(xx * cc));
            mean += distances[word];
         }
         mean /= (float)(settings.numberVisualWords);
         for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
            activations[word] += ( mean - distances[word]> 0.0 ? mean - distances[word] : 0.0);
         }
			
      } else if(settings.simFunction == SOFT_ASSIGNMENT_CLIPPING){
         for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
            cc = 0.0;
            xc = 0.0;
            
            for(size_t dim = 0; dim < dataDims; ++dim){
               cc += bow[word].content[dim] * bow[word].content[dim];
               xc += bow[word].content[dim] * features[feat].content[dim];
               
            }
            
            
            distances[word] = sqrt(cc + (xx - (2 * xc))) ;
            //distances[word] = sqrt(dist);
            
            mean += distances[word];
         }
         mean /= (float)(settings.numberVisualWords);
         for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
            activations[word] += ( mean - distances[word]> 0.0 ? mean - distances[word] : 0.0);
         }
         
         //and again!
         mean = 0;
         for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
            mean += activations[word];            
         }
         
         mean /= settings.numberVisualWords;
         
         for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
            activations[word] = (activations[word] - mean) > 0.0 ? (activations[word] - mean) : 0.0;
         }
         
      }

   }
   //standardize(activations);
   return activations;
}

vector< float > Codebook::getQActivations(vector<Feature> features){
   
    unsigned int nQuadrants = 4; //must be a square!
    
   vector< float> activations(settings.numberVisualWords * nQuadrants, 0.0);
   unsigned int dataDims = features[0].content.size();
   vector<float> distances(settings.numberVisualWords);
   float dev;
   float mean;
   
   float xx = 0.0;
   float cc;
   float xc;
   
   
   unsigned int sqrtQ = (unsigned int)sqrt(nQuadrants);
   unsigned int sqrtP = (unsigned int)sqrt(features.size());
   unsigned int quadSize = (unsigned int)sqrt(features.size() / nQuadrants);
   bool overlap = features.size() % nQuadrants != 0;
   
   for(size_t qIdx = 0; qIdx < nQuadrants; ++qIdx){

      unsigned int qX = qIdx % sqrtQ;
      unsigned int qY = (qIdx - qX) / sqrtQ;

      for(size_t pX = qX * quadSize; pX < (qX + 1) * quadSize + (overlap ? 1 : 0); ++pX){
         for(size_t pY = qY * quadSize; pY < (qY + 1) * quadSize + (overlap ? 1 : 0); ++pY){
         
            unsigned int pIdx = pY * sqrtP + pX;
            if(settings.standardize)
               standardize(features[pIdx].content, 10);
            //calculate activation;
            if(settings.simFunction == SOFT_ASSIGNMENT){
               xx = 0;
               for(size_t dim = 0; dim < dataDims; ++dim){
                  xx += features[pIdx].content[dim] * features[pIdx].content[dim];
               }
               
               mean = 0.0;
               for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
                  cc = 0.0;
                  xc = 0.0;

                  for(size_t dim = 0; dim < dataDims; ++dim){
                     cc += bow[word].content[dim] * bow[word].content[dim];
                     xc += bow[word].content[dim] * features[pIdx].content[dim];

                  }
                  //float dist = 0.0;
                  //for(size_t dim = 0; dim < dataDims; ++dim){
                  //   dist += (bow[cl][word].content[dim] - features[feat].content[dim]) *  (bow[cl][word].content[dim] - features[feat].content[dim]);
                  //}

                  distances[word] = sqrt(cc + (xx - (2 * xc))) ;
                  //distances[word] = sqrt(dist);

                  mean += distances[word];
               }
               
               mean /= (float)(settings.numberVisualWords);
               
               for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
                  activations[qIdx * settings.numberVisualWords + word] += ( mean - distances[word] > 0.0 ? (mean - distances[word]) : 0.0);
               }
               
            }
            else if(settings.simFunction == CB_RBF){
               for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
               
                  distances[word] = bow[word].getDistanceSq(features[pIdx]);
                  dev = exp(-1.0 * distances[word] / (settings.similaritySigma));
                  activations[qIdx * settings.numberVisualWords + word] += dev;
               
               }
            }
         }
      }
         
      /*
      //standardize data
      float mean = 0;
      float stddev = 0;
      for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
         mean += activations[qIdx * settings.numberVisualWords + word];
      }
      
      mean /= settings.numberVisualWords;
      
      for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
         stddev += (activations[qIdx * settings.numberVisualWords + word] - mean) * (activations[qIdx * settings.numberVisualWords + word] - mean);
      }
      
      stddev /= settings.numberVisualWords;
      stddev += 0.01; //no devision by zero
      stddev = sqrt(stddev);
      
      for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
         activations[qIdx * settings.numberVisualWords + word] = (activations[qIdx * settings.numberVisualWords + word] - mean) / stddev;
      }
      */
      
	
      
   }
   standardize(activations, 0.01);
   return activations;
}

void Codebook::importCodebook(string filename){
   charInt fancyInt;
   charfloat fancyfloat;
   unsigned int typesize;
   unsigned int featDims;
   
   ifstream file(filename.c_str(), ios::binary);
   
   //read number of classes
   file.read(fancyInt.chars,4);
   nClasses = fancyInt.intVal;
   if(normalOut)
      cout << "Codebook import: " << nClasses << " classes\n";
   
   //read nr of visual words
   file.read(fancyInt.chars, 4);
   settings.numberVisualWords = fancyInt.intVal;
   if(normalOut)
      cout << "Codebook import: " << settings.numberVisualWords << " words per class\n";
   //read typesize
   char c;
   file.read(&c,1);
   typesize = c;
   //let the compiler shutup about the fact that there is no dynamic type support yet
   ++typesize;
   --typesize;
   
   //read feature dimensionality
   file.read(fancyInt.chars, 4);
   
   featDims = fancyInt.intVal;
   //allocate space

   //read centroids

   for (size_t idx = 0; idx < settings.numberVisualWords; ++idx){
      //Feature f(featDims,0);
      Centroid c;
      c.content.resize(featDims);
      for(size_t featIdx = 0; featIdx < featDims; ++featIdx){
         file.read(fancyfloat.chars, 8);
         c.content[featIdx] = fancyfloat.floatVal;
      }
      bow.push_back(c);
   }
   
   
   file.close();
}

void Codebook::exportCodebook(string filename){
   /* codebook file conventions:
    * 
    * first, the number of classes(int4)
    * second one line with one number (4 bytes), representing the number of visual words
    * third, the size of the primitive-types of each value (float, float etc)  (1 byte)
    * fourth,  one number: the number of dimensions of each visual words. (4 bytes)
    * for each class 
    *    a little-endian binary dump of the visual words.
    *  
    *  No seperator characters are used
   */
   
   charInt fancyInt;
   charfloat fancyfloat;
   
   //cout << "\t\twordSize:\t" << bow[0].content.size() << "\n\tfilename:\t" << filename.c_str() << endl;
   unsigned int wordSize = bow[0].content.size();
   ofstream file(filename.c_str(),  ios::binary);
   
   //write nr of classes
   fancyInt.intVal = 1;
   file.write(fancyInt.chars, 4);
   
   //write nr visual words per class
   fancyInt.intVal = settings.numberVisualWords;
   file.write(fancyInt.chars, 4);
   
   //type size
   char c = 8;
   file.write(&c, 1);
  
   //write dimensionality of words
   fancyInt.intVal = wordSize;
   file.write(fancyInt.chars, 4);
   
   for(size_t word = 0; word < settings.numberVisualWords; ++word){
      for (size_t val = 0; val < wordSize; ++val){
         fancyfloat.floatVal = bow[word].content[val];
         file.write(fancyfloat.chars, 8);
      }
   } 
   
   file.close();
}
