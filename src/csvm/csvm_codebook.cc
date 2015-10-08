#include <csvm/csvm_codebook.h>

using namespace std;
using namespace csvm;

void Codebook::constructCodebook(vector<Feature> featureset){
   settings.method = /*KMeans_Clustering;*/LVQ_Clustering;
   settings.numberVisualWords = 300;
   cout << "clustering " << featureset.size() << " 1 x " << featureset[0].content.size() <<" features..\n";
   switch(settings.method){
      case LVQ_Clustering:
         bow = lvq.cluster(featureset, settings.numberVisualWords, 0.05,10);
         break;
      case KMeans_Clustering:
         bow = kmeans.cluster(featureset, 8);
         break;
   }
   
}

Feature Codebook::getActivations(Feature* f){
   Feature act(settings.numberVisualWords,0);
   vector<double> distances(settings.numberVisualWords);
   double meanDist = 0;
   double dev;
   
   for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
//       distances[word] = sqrt(bow[word].getDistanceSq(f));
      distances[word] = bow[word].getManhDist(f);
      meanDist += distances[word];
   }
   meanDist /= (double)settings.numberVisualWords;
   
   for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
      //dev = meanDist - distances[word];
      dev = 1 - distances[word] / meanDist;
      act.content[word] = dev > 0 ? dev : 0;
      //out << act.content[word] << endl;
   }
       
   act.label = f->label;
   
   return act;
}

void Codebook::importCodebook(string filename){
   charInt fancyInt;
   charDouble fancyDouble;
   unsigned int typesize;
   unsigned int featDims;
   
   ifstream file(filename.c_str(), ios::in | ios::binary);
   //read first int (nVisualWords)
   for(size_t idx = 0; idx < 4; ++idx){
      file >> fancyInt.chars[idx];
   }
   settings.numberVisualWords = fancyInt.intVal;
   
   //read typesize
   file >> typesize;
   
   //read feature dimensionality
   for(size_t idx = 0; idx < 4; ++idx){
      file >> fancyInt.chars[idx];
   }
   featDims = fancyInt.intVal;
   //allocate space
   
   bow.reserve(settings.numberVisualWords);
   //read centroids
   
   for (size_t idx = 0; idx < settings.numberVisualWords; ++idx){
      Feature f(featDims,0);
      for(size_t featIdx = 0; featIdx < featDims; ++featIdx){
         for(size_t doubleIdx = 0; doubleIdx < 8; ++doubleIdx)
            file >> fancyDouble.chars[doubleIdx];
         f.content[featIdx] = fancyDouble.doubleVal;
         
      }
   }
   
   file.close();
}

void Codebook::exportCodebook(string filename){
   /* codebook file conventions:
      first one line with one number (4 bytes), representing the number of visual words
      second, the size of the primitive-types of each value (double, float etc)  (1 byte)
      third, one line with one number: the number of dimensions of each visual words. (4 bytes)
      fourth a little-endian binary dump of the visual words.
      
      No seperator characters are used
   */
   
   charInt fancyInt;
   charDouble fancyDouble;
   
   
   
   unsigned int wordSize = bow[0].content.size();
   
   ofstream file(filename.c_str(), ios::out | ios::binary);
   
   //write nr visual words
   fancyInt.intVal = settings.numberVisualWords;
   file << fancyInt.chars[0] << fancyInt.chars[1] << fancyInt.chars[2] << fancyInt.chars[3];
   //type size
   file << (unsigned char) 8;
   
   //dimensionality of features
   
   fancyInt.intVal = wordSize;
   file << fancyInt.chars[0] << fancyInt.chars[1] << fancyInt.chars[2] << fancyInt.chars[3];
   
   for(size_t word = 0; word < settings.numberVisualWords; ++word){
      for (size_t val = 0; val < wordSize; ++val){
         fancyDouble.doubleVal = bow[word].content[val];
         file << fancyDouble.chars[0] << fancyDouble.chars[1] << fancyDouble.chars[2] << fancyDouble.chars[3] << fancyDouble.chars[4] << fancyDouble.chars[5] << fancyDouble.chars[6] << fancyDouble.chars[7];
      }
   } 
   
   file.close();
}