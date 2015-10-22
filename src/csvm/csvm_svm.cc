#include <csvm/csvm_svm.h>

using namespace std;
using namespace csvm;


SVM::SVM(int datasetSize, int nClusters, int nCentroids, unsigned int labelId){
   
   alphaData = vector<double>(datasetSize,1.0 / datasetSize);
   //cout << "alphaData =  " << alphaData[0] << endl;
   alphaCentroids = vector < vector<double> >(nClusters, vector<double>(nCentroids,0.01f));
   //cout << "alphaCentr =  " << alphaCentroids[0][0] << endl;

   this->classId = labelId;

   
   bias = 0;
}

void SVM::setSettings(SVM_Settings s){
   settings = s;
}

double SVM::updateAlphaData(vector<Feature>& clActivations, unsigned int dataIdx){
   unsigned int dataLabel = clActivations[0].getLabelId();
   double diff = 0.0;
   double target;
   unsigned int nClasses = clActivations.size();
   unsigned int nCentroids = clActivations[0].content.size();
   double sum = 0.0;
   double yData = (dataLabel == classId ? 1.0 : -1.0);
   double yCentroid;
   for(size_t cl = 0; cl < nClasses; ++cl){
      yCentroid = (cl == classId ? 1.0 : -1.0);
      for(size_t centr = 0; centr < nCentroids; ++centr){
         sum += alphaData[dataIdx] * yData * yCentroid * clActivations[cl].content[centr];
         //cout << "diff = " << clActivations[cl].content[centr] << endl;
      }
   }
   sum = -1.0 * sum;
   target = alphaData[dataIdx] + (settings.learningRate * sum);
   //cout << "target = " << target << endl;
   target = target < 0.0 ? 0.0 : target;
   target = target > settings.SVM_C_Data ? settings.SVM_C_Data : target;
   diff = alphaData[dataIdx] - target;
   alphaData[dataIdx] = target;
   return diff;
}

void SVM::constrainAlphaCentroid(vector< vector< Feature > >& activations, unsigned int nIterations, double cost){
   double sum;
   unsigned int nClasses = alphaCentroids.size();
   unsigned int nCentroids = alphaCentroids[0].size();
   double yData;
   double oldVal;
   double deltaAlpha;
   double target;
   for(size_t iter = 0; iter < nIterations; ++iter){
      sum = 0.0;
      for(size_t classIdx0 = 0; classIdx0 < nClasses; ++classIdx0){
         for(size_t centrIdx = 0; centrIdx < nCentroids; ++ centrIdx){
            yData = classIdx0 == classId ? 1.0 : -1.0;
            sum += alphaCentroids[classIdx0][centrIdx] * yData;
            //cout << "sum = " << alphaData[dIdx0] << endl;
         //getchar();
         }
      }
      
      for(size_t classIdx0 = 0; classIdx0 < nClasses; ++classIdx0){
         for(size_t centrIdx = 0; centrIdx < nCentroids; ++ centrIdx){
            yData = classIdx0 == classId ? 1.0 : -1.0;
            oldVal = alphaCentroids[classIdx0][centrIdx];
            deltaAlpha = -2.0 * cost * sum * yData;
            target = alphaCentroids[classIdx0][centrIdx] + deltaAlpha * settings.learningRate;
            //cout << "constrain target: " << target << endl;
            target = target < 0.0 ? 0.0 : target;
            target = target > settings.SVM_C_Centroid ? settings.SVM_C_Centroid : target;
            
            alphaCentroids[classIdx0][centrIdx] = target;
            sum += ( alphaCentroids[classIdx0][centrIdx] - oldVal ) * yData;
            //getchar();
         }
      }
      
   }
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
         //cout << "update alphaData with label == " << ((unsigned int)(activations[dIdx0][0].getLabelId())) << endl;
         yData = ((unsigned int)(activations[dIdx0][0].getLabelId())) == classId ? 1.0 : -1.0;
         sum += alphaData[dIdx0] * yData;
         //cout << "sum = " << alphaData[dIdx0] << endl;
        // getchar();
      }
      
      for(size_t dIdx1 = 0; dIdx1 < size; ++dIdx1){
         yData = ((unsigned int)(activations[dIdx1][0].getLabelId())) == classId ? 1.0 : -1.0;
         oldVal = alphaData[dIdx1];
         deltaAlpha = -2.0 * cost * sum * yData;
         target = alphaData[dIdx1] + deltaAlpha * settings.learningRate;
         //cout << "constrain target: " << target << endl;
         target = target < 0.0 ? 0.0 : target;
         target = target > settings.SVM_C_Data ? settings.SVM_C_Data : target;
         
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
   target = alphaCentroids[centrClass][centr] + settings.learningRate * ((double)1.0 - ( (double)sum));
   target = target < 0.0 ? 0.0 : target;
   target = target > settings.SVM_C_Centroid ? settings.SVM_C_Centroid : target;
   diff = alphaCentroids[centrClass][centr] - target;
   //cout << "centroid alpha target = " << target << ", which has a difference of " << diff << endl;
   alphaCentroids[centrClass][centr] = target;
   return diff;
}


double SVM::updateAlphaDataClassic(vector< Feature > simKernel, CSVMDataset* ds, double D2){
   
   double diff = 0.0;
   double deltaDiff = 0.0;
   double target;
   double sum = 0.0;
   double yData0;
   double yData1;
   unsigned int nData = simKernel.size();
   double deltaAlpha;

   
   for(size_t dIdx0 = 0; dIdx0 < nData; ++dIdx0){
      deltaDiff = 0.0;
      yData0 = ((unsigned int)(ds->getImagePtr(dIdx0)->getLabelId()) == classId ? 1.0 : -1.0);
      for(size_t dIdx1 = 0; dIdx1 < nData; ++dIdx1){
         yData1 = ((unsigned int)(ds->getImagePtr(dIdx1)->getLabelId()) == classId ? 1.0 : -1.0);
         //cout << "yData0 = " << yData0 << ", Ydata1 = " << yData1 << endl;
         //cout << "Similarity = " << simKernel[dIdx0].content[dIdx1] << endl; 
         sum += alphaData[dIdx1] * yData0 * yData1 * simKernel[dIdx0].content[dIdx1];
      }
      deltaAlpha = 1.0 - sum;
      target = D2 * alphaData[dIdx0] + deltaAlpha * settings.learningRate;
      target = target > settings.SVM_C_Data ? settings.SVM_C_Data : target;
      target = target < 0.0 ? 0.0 : target;
      deltaDiff = alphaData[dIdx0] - target;
      diff += (deltaDiff < 0.0 ? deltaDiff * -1.0 : deltaDiff);
      alphaData[dIdx0] = target;
      
   }
   
   return diff;
}

double SVM::constrainAlphaDataClassic(vector< Feature > simKernel, CSVMDataset* ds, double cost, unsigned int nIterations){
   
   double sum = 0;
   double oldVal;
   unsigned int nData = simKernel.size();
   double deltaAlpha;
   double yData; 
   double target;
   
   double diff = 0.0;
   double deltaDiff = 0.0;
   
   for(size_t constItr = 0; constItr < nIterations; ++constItr){
      for(size_t dIdx0 = 0; dIdx0  < nData; ++dIdx0){
         yData = (classId == (unsigned int)(ds->getImagePtr(dIdx0)->getLabelId()) ? 1.0 : -1.0);
         sum += alphaData[dIdx0] * yData;
      }
      
      for(size_t dIdx0 = 0; dIdx0  < nData; ++dIdx0){
         yData = (classId == (unsigned int)(ds->getImagePtr(dIdx0)->getLabelId()) ? 1.0 : -1.0);
         //cout << "yData = " << yData << endl;
         oldVal = alphaData[dIdx0];
         deltaAlpha = -2.0 * cost * sum * yData;
         target = alphaData[dIdx0] + deltaAlpha * settings.learningRate;
         target = target > settings.SVM_C_Data ? settings.SVM_C_Data : target;
         target = target < 0.0 ? 0.0 : target;
         deltaDiff = alphaData[dIdx0] - target;
         diff += (deltaDiff < 0.0 ? deltaDiff * -1.0 : deltaDiff);
         
         alphaData[dIdx0] = target;
         sum += (alphaData[dIdx0] - oldVal) * yData;
      }
   }
   return diff;
}

void SVM::calculateBiasClassic(vector<Feature> simKernel, CSVMDataset* ds){
   bias = 0;
   unsigned int total = 0;
   unsigned int nData = simKernel.size();
   double output;
   double yData;
   
   for(size_t dIdx0 = 0; dIdx0 < nData; ++dIdx0){
      //if((alpha_coeff[C][i] > 0.0) && ((alpha_coeff[C][i]) < (SVM_C - 0.000001)  Marco 
      if((alphaData[dIdx0] > 0) && (alphaData[dIdx0] < settings.SVM_C_Data - 0.000001)){
         output = 0;
         for(size_t dIdx1 = 0; dIdx1 < nData; ++dIdx1){
            yData = ((unsigned int)(ds->getImagePtr(dIdx1)->getLabelId()) == classId ? 1.00 : -1.00);
            output += alphaData[dIdx1] * yData * simKernel[dIdx0].content[dIdx1];
         }
         yData = ((unsigned int)(ds->getImagePtr(dIdx0)->getLabelId()) == classId ? 1.00 : -1.00);
         bias += yData * output;
         ++total;
      }
   }
   
   if(total == 0)
      bias = 0;
   else 
      bias /= total;
}
void SVM::trainClassic(vector<Feature> simKernel, CSVMDataset* ds){
   
   double sumDeltaAlpha = 1000.0;
   double prevSumDeltaAlpha = 100.0;
   double deltaAlphaData;
   double convergenceThreshold = 10 * settings.learningRate;
   for(size_t round = 0; abs(prevSumDeltaAlpha -sumDeltaAlpha) > convergenceThreshold; ++round){
      prevSumDeltaAlpha = sumDeltaAlpha;
      sumDeltaAlpha = 0.0;
      deltaAlphaData = updateAlphaDataClassic(simKernel, ds,1);
      deltaAlphaData = deltaAlphaData < 0.0 ? deltaAlphaData * -1.0 : deltaAlphaData;
      sumDeltaAlpha += deltaAlphaData;
      
      constrainAlphaDataClassic(simKernel, ds, 1, 4 );
      cout << "SVM " << classId << " training round " << round << ".  Sum of Change  = " << fixed << sumDeltaAlpha << " DeltaSOC = " << (prevSumDeltaAlpha - sumDeltaAlpha) << endl;
   }
   calculateBiasClassic(simKernel, ds);
}



void SVM::train(vector< vector<Feature> >& activations){
   unsigned int nClasses = activations[0].size();
   unsigned int nCentroids = activations[0][0].content.size();
   //cout << "I'm reading " << nCentroids << " centroids, and " << nClasses << " classes\n";
   //cout << "[1,1] act = " << activations[0][0].content[0] << endl;
   unsigned int size = activations.size();
   double sumDeltaAlpha = 1000.0f;

   double prevSumDeltaAlpha = 10000.0f;
   double deltaAlphaData, deltaAlphaCentroid;
   double convergenceThreshold = 0.00005;
   cout << "Yay! I'm a learning SVM, learing on " << size << " data\n";
   while(((prevSumDeltaAlpha - sumDeltaAlpha) > 0 ? (prevSumDeltaAlpha - sumDeltaAlpha) : (prevSumDeltaAlpha - sumDeltaAlpha) * -1 ) > convergenceThreshold){
      
      
      
      prevSumDeltaAlpha = sumDeltaAlpha;
      sumDeltaAlpha = 0.0;
      for(size_t dataIdx = 0; dataIdx < size; ++dataIdx){
         deltaAlphaData = updateAlphaData(activations[dataIdx], dataIdx);
         deltaAlphaData = deltaAlphaData < 0.0 ? deltaAlphaData * -1.0 : deltaAlphaData;
         sumDeltaAlpha += deltaAlphaData;
      }

      contstrainAlphaData(activations, 4, 1);
      
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
      constrainAlphaCentroid(activations, 4,1);
      
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

double SVM::classifyClassic(vector<Feature> f, vector< vector<Feature> > datasetActivations, CSVMDataset* ds){
   double result = 0;
   unsigned int nData = datasetActivations.size();
  
   double yData;
   double kernel = 0.0;
   double sigma = .5;
   unsigned int nClasses = datasetActivations[0].size();
   unsigned int nCentroids = datasetActivations[0][0].content.size();
   Feature dataKernel(nData,0.0);
   
   for(size_t dIdx0 = 0; dIdx0 < nData; ++dIdx0){
   
      double sum = 0;
      for(size_t cl = 0; cl < nClasses; ++cl){
         for(size_t centr = 0; centr < nCentroids; ++centr){
            sum += (datasetActivations[dIdx0][cl].content[centr] - f[cl].content[centr])*(datasetActivations[dIdx0][cl].content[centr] - f[cl].content[centr]);
         }
      }
      
      
     kernel = exp((-1.0*sqrt(sum))/settings.sigmaClassicSimilarity);
     yData = (unsigned int)(ds->getImagePtr(dIdx0)->getLabelId()) == classId ? 1.0 : -1.0;
     result += alphaData[dIdx0] * yData * kernel;
      
   }
   
   result += bias;
   return result;
}