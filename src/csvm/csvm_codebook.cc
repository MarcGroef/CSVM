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
   vector<Feature> activations(nClasses,Feature(settings.numberVisualWords, 0));
   
   vector<double> distances(settings.numberVisualWords);
   double dev;
   unsigned int nFeatures = features.size();
   
   for(size_t feat = 0; feat < nFeatures; ++feat){
      for(size_t cl = 0; cl < nClasses; ++cl){
         
         
         for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
            distances[word] = sqrt(bow[cl][word].getDistanceSq(features[feat]));
         }
         
         for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
            
            dev = exp(-1.0 * distances[word] / settings.similaritySigma);
            activations[cl].content[word] += dev;
         
         }
         activations[cl].label = features[feat].label;
         activations[cl].labelId = features[feat].labelId;
      }
   }
   
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