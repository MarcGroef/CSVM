#include <csvm/csvm_svm.h>

using namespace std;
using namespace csvm;


SVM::SVM(int datasetSize, int nClusters, double learningRate, unsigned int labelId, int dataDims){
   alphaData = vector<double>(datasetSize,0.5f);
   alphaCentroids = vector < vector<double> >(nClusters, vector<double>(nClusters,0.5f));
   this->learningRate = learningRate;
   this->classId = labelId;
   finalDataWeights = vector <double> (dataDims,0);
   this->dataDims = dataDims;
   
   
}

double SVM::updateAlphaData(vector<Feature> clActivations, unsigned int dataIdx){
   unsigned int dataLabel = clActivations[0].getLabelId();
   double diff = 0;
   unsigned int nClasses = clActivations.size();
   unsigned int nCentroids = clActivations[0].content.size();
   
   int yData = (dataLabel == classId ? 1 : -1);
   int yCentroid;
   for(size_t cl = 0; cl < nClasses; ++cl){
      yCentroid = (cl == classId ? 1 : -1);
      for(size_t centr = 0; centr < nCentroids; ++centr){
         diff += alphaData[dataIdx] * yData * yCentroid * clActivations[cl].content[centr];
      }
   }
   return learningRate * (-0.5 * diff);
}

double SVM::updateAlphaCentroid(vector< vector< Feature> > clActivations, unsigned int centrClass, int centr){
   double sum = 0;
   unsigned int nData = clActivations.size();
   unsigned int yData, yCentroid;
   
   
   yCentroid = (centrClass == classId ? 1 : -1);
   
   for(size_t dataIdx = 0; dataIdx < nData; ++dataIdx){
      yData = ((unsigned int)(clActivations[dataIdx][0].getLabelId()) == classId ? 1 : -1);
      sum += alphaData[dataIdx] * yData * yCentroid * clActivations[dataIdx][centrClass].content[centr];
   }
   
   return 1 - 0.5 * sum;
}


void SVM::train(vector< vector<Feature> >activations){
   unsigned int nClasses = activations[0].size();
   unsigned int nCentroids = activations[0][0].content.size();
   //cout << "I'm reading " << nCentroids << " centroids\n";
   unsigned int size = activations.size();
   double sumDeltaAlpha = 1;
   double prevSumDeltaAlpha = 1;
   double deltaAlphaData, deltaAlphaCentroid;
   double convergenceThreshold = 0.1;
   //cout << "Yay! I'm a learning SVM!\n";
   while(sumDeltaAlpha > convergenceThreshold){
      cout << "Yay, SVM training iteration round! Delta = " << sumDeltaAlpha << " \n";
      sumDeltaAlpha = 0;
      for(size_t dataIdx = 0; dataIdx < size; ++dataIdx){
         //cout << "update alphaData's\n";
         deltaAlphaData = updateAlphaData(activations[dataIdx], dataIdx);
         sumDeltaAlpha += deltaAlphaData;
         alphaData[dataIdx] += deltaAlphaData;
      }
      for(size_t cl = 0; cl < nClasses; ++cl){
         //cout << "update alhpaCentroid class " << cl << endl;
         for(size_t centr = 0; centr < nCentroids; ++centr){
            //cout << "update alhpaCentroid centrIdx " << centr << endl;
            deltaAlphaCentroid = updateAlphaCentroid(activations, cl, centr);
            alphaCentroids[cl][centr] += deltaAlphaCentroid;
         }
      }
      
   }
 
}

int SVM::classify(vector<Feature> f, Codebook* cb){
   double result = 0;
   unsigned int nClasses = cb->getNClasses();
   unsigned int nCentroids = cb->getNCentroids();
   int yCentroid;
   
   for(size_t cl = 0; cl < nClasses; ++cl){
      yCentroid = (cl == classId ? 1 : -1);
      for(size_t centr = 0; centr < nCentroids; ++centr){
         result += alphaCentroids[cl][centr] * yCentroid * f[cl].content[centr];
      }
   }
   return result;
}
