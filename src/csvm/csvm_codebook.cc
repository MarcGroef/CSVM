#include <csvm/csvm_codebook.h>

using namespace std;
using namespace csvm;

union charInt{
   char chars[4];
   unsigned int intVal;
};

union charDouble{
   char chars[8];
   double doubleVal;
};

Codebook::Codebook(){
   nClasses = 10;
   bow.resize(nClasses);
}

void Codebook::setSettings(Codebook_settings s){
   settings = s;
}
Feature Codebook::getCentroid(int cl, int centrIdx){
   return bow[cl][centrIdx];
}

void Codebook::constructCodebook(vector<Feature> featureset,int labelId){
   cout << "constructing codebook for label " << labelId << " in ";
   switch(settings.method){
      case LVQ_Clustering:
         bow[labelId] = lvq.cluster(featureset, labelId, settings.numberVisualWords, 0.1,120);
         break;
      case KMeans_Clustering:
         bow[labelId] = kmeans.cluster(featureset, settings.numberVisualWords);
         break;
   }
   
}

unsigned int Codebook::getNClasses(){
   
   
   return nClasses;
}


unsigned int Codebook::getNCentroids(){
   return settings.numberVisualWords;
}


vector<Feature> Codebook::getActivations(vector<Feature> features){
   vector<Feature> activations(nClasses,Feature(settings.numberVisualWords, 0.0));
   vector<Feature> activation(nClasses,Feature(settings.numberVisualWords, 0.0));
   unsigned int dataDims = features[0].content.size();
   vector<double> distances(settings.numberVisualWords);
   double dev;
   unsigned int nFeatures = features.size();
   double totDist = 0;
   double classDist;
   double mean = 0;
   
   double xx;
   double cc;
   double xc;
   
   for(size_t feat = 0; feat < nFeatures; ++feat){
      totDist = 0;
      //cout << "ACTIVATIONS FEATURE " << feat << ":\n";
      xx = 0.0;
      
      for(size_t dim = 0; dim < dataDims; ++dim){
         xx += features[feat].content[dim] * features[feat].content[dim];
      }
      for(size_t cl = 0; cl < nClasses; ++cl){
         classDist = 0;
         mean = 0.0;
         //cout << "cl" << cl << ": ";
         /*for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
            distances[word] = /*sqrt(//bow[cl][word].getDistanceSq(features[feat]));
            mean += distances[word];
         }
         mean /= (double)(settings.numberVisualWords);
         
         for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
            //cout << "distances class " << cl << ", word: " << word << " = " << distances[word] << endl;
            //dev = exp(-1.0 * distances[word] / (2 * settings.similaritySigma * settings.similaritySigma));
            //activations[cl].content[word] += dev;
            dev = mean - distances[word];
            activations[cl].content[word] += (dev > 0.0 ? dev : 0.0);
            //cout << "activation word " << word << " is : " << dev << endl;
            
         }*/
         
         //As done by Ng:
         
         
         
         for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
            cc = 0.0;
            xc = 0.0;
            for(size_t dim = 0; dim < dataDims; ++dim){
               cc += bow[cl][word].content[dim] * bow[cl][word].content[dim];
               xc += bow[cl][word].content[dim] * features[feat].content[dim];
            }
            
            distances[word] = sqrt(cc + (xx - (2 * xc))) ;
            mean += distances[word];
         }
         mean /= (double)(settings.numberVisualWords);
         
         for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
            //activation[cl].content[word] += ( mean - distances[word] > 0.0 ? mean - distances[word] : 0.0);
            activations[cl].content[word] += ( mean - distances[word]> 0.0 ? mean - distances[word] : 0.0);
         }
         
         //cout << endl;
         activation[cl].label = features[feat].label;
         activation[cl].labelId = features[feat].labelId;
         //cout << "totDist class " << cl << " is : " << classDist << endl;
      }
      //normalize activations
      /*for(size_t cl = 0; cl < nClasses; ++cl){
         mean = 0.0;
         dev = 0.0;
         for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
            mean += activation[cl].content[word];
         }

         mean /= (double)(nClasses * settings.numberVisualWords);
         
         for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
            dev += (mean - activation[cl].content[word]) * (mean - activation[cl].content[word]);
         }
         
         dev = sqrt(dev) + 0.001;

         for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
            activations[cl].content[word] += (activation[cl].content[word] - mean) / dev;
         }
      }*/
   }
   
   //normalize activation summation
   /*for(size_t cl = 0; cl < nClasses; ++cl){
      mean = 0;
      dev = 0;
      for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
         mean += activations[cl].content[word];
      }

      mean /= (double)(nClasses * settings.numberVisualWords);
      
      for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
         dev += (mean - activations[cl].content[word]) * (mean - activations[cl].content[word]);
      }
      
      dev = sqrt(dev);

      for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
         activations[cl].content[word] += (activations[cl].content[word] - mean) / dev;
      }
   }*/
   
   
   
   
   return activations;
}

void Codebook::importCodebook(string filename){
   charInt fancyInt;
   charDouble fancyDouble;
   unsigned int typesize;
   unsigned int featDims;
   
   ifstream file(filename.c_str(), ios::binary);
   
   //read number of classes
   file.read(fancyInt.chars,4);
   nClasses = fancyInt.intVal;

   //read nr of visual words
   file.read(fancyInt.chars, 4);
   settings.numberVisualWords = fancyInt.intVal;
   
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
   bow.clear();
   bow.resize(nClasses);
   //read centroids
   for(size_t cl = 0; cl < nClasses; ++cl){
      for (size_t idx = 0; idx < settings.numberVisualWords; ++idx){
         Feature f(featDims,0);
         for(size_t featIdx = 0; featIdx < featDims; ++featIdx){
            file.read(fancyDouble.chars, 8);
            f.content[featIdx] = fancyDouble.doubleVal;
         }
         bow[cl].push_back(f);
      }
   }
   
   file.close();
}

void Codebook::exportCodebook(string filename){
   /* codebook file conventions:
    * 
    * first, the number of classes(int4)
    * second one line with one number (4 bytes), representing the number of visual words
    * third, the size of the primitive-types of each value (double, float etc)  (1 byte)
    * fourth,  one number: the number of dimensions of each visual words. (4 bytes)
    * for each class 
    *    a little-endian binary dump of the visual words.
    *  
    *  No seperator characters are used
   */
   
   charInt fancyInt;
   charDouble fancyDouble;

   
   unsigned int wordSize = bow[0][0].content.size();
   ofstream file(filename.c_str(),  ios::binary);
   
   //write nr of classes
   fancyInt.intVal =(int)nClasses;
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
   
   for(size_t cl = 0; cl < nClasses; ++cl){
      for(size_t word = 0; word < settings.numberVisualWords; ++word){
         for (size_t val = 0; val < wordSize; ++val){
            fancyDouble.doubleVal = bow[cl][word].content[val];
            file.write(fancyDouble.chars, 8);
         }
      } 
   }
   
   file.close();
}