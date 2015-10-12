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

Feature Codebook::getCentroid(int cl, int centrIdx){
   return bow[cl][centrIdx];
}

void Codebook::constructCodebook(vector<Feature> featureset,int labelId){
   settings.method = /*KMeans_Clustering;*/LVQ_Clustering;
   settings.numberVisualWords = 300;
   switch(settings.method){
      case LVQ_Clustering:
         bow[labelId] = lvq.cluster(featureset, settings.numberVisualWords, 0.1,120);
         break;
      case KMeans_Clustering:
         bow[labelId] = kmeans.cluster(featureset, 8);
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
   double meanDist = 0;
   double dev;
   Feature* f = &features[0];
   //cout << "bow has [" << bow.size() << "][" << bow[0].size() << "][" << bow[0][0].content.size() << "]\n";
   for(size_t cl = 0; cl < nClasses; ++cl){
      f = &features[cl];
      for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
         distances[word] = bow[cl][word].getDistanceSq(f);
         meanDist += distances[word];
      }
      meanDist /= (double)settings.numberVisualWords;
      
      for(unsigned int word = 0; word < settings.numberVisualWords; ++word){
         //dev = meanDist - distances[word];
         dev = meanDist - distances[word];
         activations[cl].content[word] += dev > 0 ? dev : 0;
         //out << act.content[word] << endl;
      }
      activations[cl].label = f->label;
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
   //for(size_t idx = 0; idx < 4; ++idx){
      file.read(fancyInt.chars,4);
   //}
   
   nClasses = fancyInt.intVal;
   cout << "reading for " << nClasses << " classes\n";
   //read first int (nVisualWords)
   //for(size_t idx = 0; idx < 4; ++idx){
      file.read(fancyInt.chars, 4);
   //}
   settings.numberVisualWords = fancyInt.intVal;
   cout << "reading for " << settings.numberVisualWords << " words each class\n";
   //read typesize
   //file >> typesize;
   char c;
   file.read(&c,1);
   typesize = c;
   cout << "read typesize = " << typesize << endl;
   //read feature dimensionality
   //for(size_t idx = 0; idx < 4; ++idx){
      file.read(fancyInt.chars, 4);
   //}
   featDims = fancyInt.intVal;
   cout << "reading for " << featDims << " dimensions\n";
   //allocate space
   bow.clear();
   bow.resize(nClasses);
   //read centroids
   for(size_t cl = 0; cl < nClasses; ++cl){
      for (size_t idx = 0; idx < settings.numberVisualWords; ++idx){
         Feature f(featDims,0);
         for(size_t featIdx = 0; featIdx < featDims; ++featIdx){
            //for(size_t doubleIdx = 0; doubleIdx < 8; ++doubleIdx)
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
   cout << "wordsize = " << wordSize << endl;
   ofstream file(filename.c_str(),  ios::binary);
   
   
   fancyInt.intVal =(int)nClasses;
   file.write(fancyInt.chars, 4);
   //file << fancyInt.chars[0] << fancyInt.chars[1] << fancyInt.chars[2] << fancyInt.chars[3];
   

   //write nr visual words
   fancyInt.intVal = settings.numberVisualWords;
   file.write(fancyInt.chars, 4);
   //file << fancyInt.chars[0] << fancyInt.chars[1] << fancyInt.chars[2] << fancyInt.chars[3];
   //type size
   char c = 8;
   file.write(&c, 1);
   //file << (unsigned char) 8;
   cout << "write typesize = " << 8 << endl;
   //dimensionality of features
   
   fancyInt.intVal = wordSize;
   //file << fancyInt.chars[0] << fancyInt.chars[1] << fancyInt.chars[2] << fancyInt.chars[3];
   file.write(fancyInt.chars, 4);
   cout << "I just wrote wordSize = " << wordSize << endl;
   for(size_t cl = 0; cl < nClasses; ++cl){
      //features
      for(size_t word = 0; word < settings.numberVisualWords; ++word){
         for (size_t val = 0; val < wordSize; ++val){
            fancyDouble.doubleVal = bow[cl][word].content[val];
            file.write(fancyDouble.chars, 8);
            //file << fancyDouble.chars[0] << fancyDouble.chars[1] << fancyDouble.chars[2] << fancyDouble.chars[3] << fancyDouble.chars[4] << fancyDouble.chars[5] << fancyDouble.chars[6] << fancyDouble.chars[7];
         }
      } 
   }
   
   file.close();
}