#include <csvm/csvm_svm.h>

using namespace std;
using namespace csvm;


SVM::SVM(int datasetSize, int nClusters, int nCentroids, double learningRate, unsigned int labelId, int dataDims){
   
   alphaData = vector<double>(datasetSize,0.5f);
   //cout << "alphaData =  " << alphaData[0] << endl;
   alphaCentroids = vector < vector<double> >(nClusters, vector<double>(nCentroids,0.5f));
   //cout << "alphaCentr =  " << alphaCentroids[0][0] << endl;
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
         //cout << "diff = " << clActivations[cl].content[centr] << endl;
      }
   }
  
   return learningRate * (-0.5 * diff);
}

//activations are just used to check labels here.
void SVM::contstrainAlphaData(vector< vector< Feature > > activations, unsigned int nIterations, double cost, double maxAlphaVal){
   double sum;
   unsigned int size = alphaData.size();
   int yData;
   double oldVal;
   double deltaAlpha;
   double target;
   for(size_t iter = 0; iter < nIterations; ++iter){
      sum = 0;
      for(size_t dIdx0 = 0; dIdx0 < size; ++dIdx0){
         yData = (unsigned int)(activations[dIdx0][0].getLabelId()) == classId ? 1 : -1;
         sum += alphaData[dIdx0] * yData;
         for(size_t dIdx1 = 0; dIdx1 < size; ++dIdx1){
            oldVal = alphaData[dIdx1];
            deltaAlpha = -2 * cost * sum * yData;
            target = alphaData[dIdx1] + deltaAlpha * learningRate;
            target = target < 0 ? 0 : target;
            target = target > maxAlphaVal ? maxAlphaVal : target;
            sum += ( alphaData[dIdx1] - oldVal ) * yData;
         }
      }
   }
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
   //sum =  
   //cout << ((0.5 * (double)sum)) << endl;
   return learningRate * ((double)1 - (0.5 * (double)sum));
}


void SVM::train(vector< vector<Feature> >activations){
   unsigned int nClasses = activations[0].size();
   unsigned int nCentroids = activations[0][0].content.size();
   //cout << "I'm reading " << nCentroids << " centroids, and " << nClasses << " classes\n";
   //cout << "[1,1] act = " << activations[0][0].content[0] << endl;
   unsigned int size = activations.size();
   double sumDeltaAlpha = 1.0f;

   double prevSumDeltaAlpha = 1.0f;
   double deltaAlphaData, deltaAlphaCentroid;
   double convergenceThreshold = 0.1;
   cout << "Yay! I'm a learning SVM, learing on " << size << " data\n";
   while(sumDeltaAlpha > convergenceThreshold){
      
      prevSumDeltaAlpha = sumDeltaAlpha;
      cout << "Yay, SVM training iteration round! Sum of Change  = " << sumDeltaAlpha << " DeltaSOC = " << abs(prevSumDeltaAlpha - sumDeltaAlpha) << endl;
      sumDeltaAlpha = 0.0;
      for(size_t dataIdx = 0; dataIdx < size; ++dataIdx){
         //cout << "update alphaData's\n";
         deltaAlphaData = updateAlphaData(activations[dataIdx], dataIdx);
         //cout << "update : " << deltaAlphaData << endl;
         sumDeltaAlpha += abs(deltaAlphaData);
         alphaData[dataIdx] += deltaAlphaData;
         //cout << "delta = " << sumDeltaAlpha << endl;
      }
      //cout << "intermediate sum delta alpha = " << sumDeltaAlpha << endl;
      contstrainAlphaData(activations, 4, 1, 1);
      
      for(size_t cl = 0; cl < nClasses; ++cl){
         //cout << "update alhpaCentroid class " << cl << endl;
         for(size_t centr = 0; centr < nCentroids; ++centr){
            //cout << "update alhpaCentroid centrIdx " << centr << endl;
            deltaAlphaCentroid = updateAlphaCentroid(activations, cl, centr);
            sumDeltaAlpha += abs(deltaAlphaCentroid);
            alphaCentroids[cl][centr] += deltaAlphaCentroid;
         }
         //cout << "intermediate sum delta alpha, @cl " << cl << " = " << sumDeltaAlpha << endl;
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
