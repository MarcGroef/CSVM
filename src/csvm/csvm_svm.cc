#include <csvm/csvm_svm.h>

using namespace std;
using namespace csvm;


SVM::SVM(int datasetSize, int nClusters, double learningRate, unsigned int labelId, int dataDims){
   alphaData = vector<double>(datasetSize,0.5f);
   alphaClusters = vector < vector<double> >(nClusters, vector<double>(nClusters,0.5f));
   this->learningRate = learningRate;
   this->classId = labelId;
   finalDataWeights = vector <double> (dataDims,0);
   this->dataDims = dataDims;
}

double SVM::updateAlphaData(vector<Feature> data, unsigned int dataIdx, Codebook* cb){
   double diff = 0;
   unsigned int nClasses = cb->getNClasses();
   int yData, yCentr;
   unsigned int nCentroids = cb->getNCentroids();
   
   for(size_t clIdx = 0; clIdx < nClasses; ++clIdx){
      yCentr = (clIdx == classId) ? 1 : -1;
      for(size_t cIdx = 0; cIdx < nCentroids; ++cIdx){
         yData = (classId == (unsigned int)(data[dataIdx].labelId)) ? 1 : -1; 
         diff += alphaClusters[clIdx][cIdx]* yData * yCentr * kernel(data[dataIdx],cb->getCentroid(clIdx, cIdx));
      }
      
   }
   return learningRate * (-0.5 * diff);
}

double SVM::kernel(Feature data, Feature centroid){  
   return sqrt(data.getDistanceSq(&centroid));
}

void SVM::train(vector<Feature> data, Codebook* cb){
   unsigned int size = data.size();
   double sumDeltaAlpha = 0;
   double prevSumDeltaAlpha = -1;
   double deltaAlpha;
   double convergenceThreshold = 0.1;
   
   while(abs(prevSumDeltaAlpha - sumDeltaAlpha > convergenceThreshold)){
   sumDeltaAlpha = 0;
      for(size_t dataIdx = 0; dataIdx < size; ++dataIdx){
         prevSumDeltaAlpha = sumDeltaAlpha;
         deltaAlpha =  updateAlphaData(data, dataIdx, cb);
         alphaData[dataIdx] += deltaAlpha;
         sumDeltaAlpha += deltaAlpha;
      }
   }
   
   for(size_t dataIdx = 0; dataIdx < size; ++dataIdx){
      for(size_t dim = 0; dim < dataDims; ++dim){
         finalDataWeights[dim] += alphaData[dataIdx] * ((unsigned int)(data[dataIdx].labelId) == classId ? 1 : -1) * data[dataIdx].content[dim];
      }
   }
}

int SVM::classify(Feature f){
   double result = 0;
   for(size_t idx = 0; idx < dataDims; ++idx){
      result += f.content[idx] * finalDataWeights[idx];
   }
   return result;
}
