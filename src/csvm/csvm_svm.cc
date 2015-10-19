#include <csvm/csvm_svm.h>

using namespace std;
using namespace csvm;


SVM::SVM(int datasetSize, int nClusters, int nCentroids, double learningRate, unsigned int labelId, int dataDims){
   
   alphaData = vector<double>(datasetSize,1.0 / datasetSize);
   //cout << "alphaData =  " << alphaData[0] << endl;
   alphaCentroids = vector < vector<double> >(nClusters, vector<double>(nCentroids,0.01f));
   //cout << "alphaCentr =  " << alphaCentroids[0][0] << endl;
   this->learningRate = learningRate;
   this->classId = labelId;
   finalDataWeights = vector <double> (dataDims,0);
   this->dataDims = dataDims;
   
   settings.SVM_C = 1.0;

}

double SVM::updateAlphaData(vector<Feature>& clActivations, unsigned int dataIdx){
   unsigned int dataLabel = clActivations[0].getLabelId();
   double diff = 0.0;
   double target;
   unsigned int nClasses = clActivations.size();
   unsigned int nCentroids = clActivations[0].content.size();
   
   double yData = (dataLabel == classId ? 1.0 : -1.0);
   double yCentroid;
   for(size_t cl = 0; cl < nClasses; ++cl){
      yCentroid = (cl == classId ? 1.0 : -1.0);
      for(size_t centr = 0; centr < nCentroids; ++centr){
         diff += alphaData[dataIdx] * yData * yCentroid * clActivations[cl].content[centr];
         //cout << "diff = " << clActivations[cl].content[centr] << endl;
      }
   }
   
   target = alphaData[dataIdx] + learningRate * ( -0.5 * diff);
   //cout << "target = " << target << endl;
   target = target < 0.0 ? 0.0 : target;
   target = target > settings.SVM_C ? settings.SVM_C : target;
   diff = alphaData[dataIdx] - target;
   alphaData[dataIdx] = target;
   return diff;
}

//activations are just used to check labels here.
void SVM::contstrainAlphaData(vector< vector< Feature > >& activations, unsigned int nIterations, double cost){
   double sum;
   unsigned int size = alphaData.size();
   double yData;
   double oldVal;
   double deltaAlpha;
   double target;
   for(size_t iter = 0; iter < nIterations; ++iter){
      sum = 0.0;
      for(size_t dIdx0 = 0; dIdx0 < size; ++dIdx0){
         yData = ((unsigned int)(activations[dIdx0][0].getLabelId())) == classId ? 1.0 : -1.0;
         sum += alphaData[dIdx0] * yData;
         //cout << "sum = " << alphaData[dIdx0] << endl;
        // getchar();
      }
      
      for(size_t dIdx1 = 0; dIdx1 < size; ++dIdx1){
         yData = ((unsigned int)(activations[dIdx1][0].getLabelId())) == classId ? 1.0 : -1.0;
         oldVal = alphaData[dIdx1];
         deltaAlpha = -2.0 * cost * sum * yData;
         target = alphaData[dIdx1] + deltaAlpha * learningRate;
         //cout << "constrain target: " << target << endl;
         target = target < 0.0 ? 0.0 : target;
         target = target > settings.SVM_C ? settings.SVM_C : target;
         
         alphaData[dIdx1] = target;
         sum += ( alphaData[dIdx1] - oldVal ) * yData;
         //getchar();
      }
      
   }
}

double SVM::updateAlphaCentroid(vector< vector< Feature> >& clActivations, unsigned int centrClass, int centr){
   double sum = 0.0;
   unsigned int nData = clActivations.size();
   double yData, yCentroid;
   double target;
   double diff;
   
   yCentroid = (centrClass == classId ? 1.0 : -1.0);
   
   for(size_t dataIdx = 0; dataIdx < nData; ++dataIdx){
      yData = ((clActivations[dataIdx][0].getLabelId()) == classId ? 1.0 : -1.0);
      //cout << "activations = " << clActivations[dataIdx][centrClass].content[centr] << endl;
      sum += alphaData[dataIdx] * yData * yCentroid * clActivations[dataIdx][centrClass].content[centr];
   }
   //cout << "sum = " << sum << endl;
   target = alphaCentroids[centrClass][centr] + learningRate * ((double)1.0 - ( (double)sum));
   target = target < 0.0 ? 0.0 : target;
   target = target > settings.SVM_C ? settings.SVM_C : target;
   diff = alphaCentroids[centrClass][centr] - target;
   //cout << "centroid alpha target = " << target << ", which has a difference of " << diff << endl;
   alphaCentroids[centrClass][centr] = target;
   return diff;
}


void SVM::train(vector< vector<Feature> >& activations){
   unsigned int nClasses = activations[0].size();
   unsigned int nCentroids = activations[0][0].content.size();
   //cout << "I'm reading " << nCentroids << " centroids, and " << nClasses << " classes\n";
   //cout << "[1,1] act = " << activations[0][0].content[0] << endl;
   unsigned int size = activations.size();
   double sumDeltaAlpha = 1.0f;

   double prevSumDeltaAlpha = 1.0f;
   double deltaAlphaData, deltaAlphaCentroid;
   double convergenceThreshold = 0.0005;
   cout << "Yay! I'm a learning SVM, learing on " << size << " data\n";
   while(sumDeltaAlpha > convergenceThreshold){
      
      
      
      prevSumDeltaAlpha = sumDeltaAlpha;
      sumDeltaAlpha = 0.0;
      /*for(size_t dataIdx = 0; dataIdx < size; ++dataIdx){
         //cout << "update alphaData's\n";
         deltaAlphaData = updateAlphaData(activations[dataIdx], dataIdx);
         //cout << "delta = " << deltaAlphaData << endl;
         deltaAlphaData = deltaAlphaData < 0.0 ? deltaAlphaData * -1.0 : deltaAlphaData;
         sumDeltaAlpha += deltaAlphaData;
         //cout << "delta = " << deltaAlphaData << endl;
      }
      //getchar();
      //cout << "intermediate sum delta alpha = " << fixed << sumDeltaAlpha << endl;
      contstrainAlphaData(activations, 4, 1);
      */
      for(size_t cl = 0; cl < nClasses; ++cl){
         //cout << "update alhpaCentroid class " << cl << endl;
         for(size_t centr = 0; centr < nCentroids; ++centr){
            //cout << "update alhpaCentroid centrIdx " << centr << endl;
            deltaAlphaCentroid = updateAlphaCentroid(activations, cl, centr);
            deltaAlphaCentroid = deltaAlphaCentroid < 0.0 ? deltaAlphaCentroid * -1.0 : deltaAlphaCentroid;
            //cout << "delta = " << deltaAlphaCentroid << endl;
            sumDeltaAlpha += deltaAlphaCentroid;
         }
         //cout << "intermediate sum delta alpha, @cl " << cl << " = " << sumDeltaAlpha << endl;
      }
      
      cout << "Yay, SVM " << classId << " training iteration round! Sum of Change  = " << fixed << sumDeltaAlpha << " DeltaSOC = " << (prevSumDeltaAlpha - sumDeltaAlpha) << endl;
      
   }
 
}

double SVM::classify(vector<Feature> f, Codebook* cb){
   double result = 0;
   unsigned int nClasses = cb->getNClasses();
   unsigned int nCentroids = cb->getNCentroids();
   double yCentroid;
   
   for(size_t cl = 0; cl < nClasses; ++cl){
      yCentroid = (cl == classId ? 1.0 : -1.0);
      for(size_t centr = 0; centr < nCentroids; ++centr){
         result += alphaCentroids[cl][centr] * yCentroid * f[cl].content[centr];
         
      }
   }
   return result;
}
