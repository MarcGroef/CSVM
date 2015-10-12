#include <csvm/csvm_svm.h>

using namespace std;
using namespace csvm;


SVM::SVM(int datasetSize, int nClusters, double learningRate){
   alphaData = vector<double>(datasetSize,0.5f);
   alphaClusters = vector < vector<double> >(nClusters, vector<double>(nClusters,0.5f));
   this-> learningRate = learningRate;
}

double SVM::updateAlphaData(vector<Feature> data, unsigned int dataIdx, Codebook* cb, unsigned int labelId){
   double diff = 0;
   unsigned int nClasses = cb->getNClasses();
   int yData, yCentr;
   unsigned int nCentroids = cb->getNCentroids();
   
   for(size_t clIdx = 0; clIdx < nClasses; ++clIdx){
      yCentr = (clIdx == labelId) ? 1 : -1;
      for(size_t cIdx = 0; cIdx < nCentroids; ++cIdx){
         yData = (labelId == (unsigned int)(data[dataIdx].labelId)) ? 1 : -1; 
         diff += alphaClusters[clIdx][cIdx]* yData * yCentr * kernel(data[dataIdx],cb->getCentroid(clIdx, cIdx));
      }
      
   }
   return learningRate * (-0.5 * diff);
}

double SVM::kernel(Feature data, Feature centroid){
   
   return 0;
}

void SVM::train(vector<Feature> data, Codebook* cb){
   unsigned int size = data.size();
   
   for(size_t dataIdx = 0; dataIdx < size; ++dataIdx){
      alphaData[dataIdx] += updateAlphaData(data, dataIdx, cb, data[dataIdx].getLabelId());
   }

}

