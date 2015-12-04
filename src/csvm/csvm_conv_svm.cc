#include<csvm/csvm_conv_svm.h>

using namespace std;
using namespace csvm;

   void ConvSVM::setSettings(ConvSVMSettings s){
      settings = s;
      weights.reserve(settings.nClasses);
      
      for(size_t clIdx = 0; clIdx < settings.nClasses; ++clIdx){
         weights[clIdx].reserve(settings.nCentroids);
         for(size_t centrIdx = 0; centrIdx < settings.nCentroids; ++centrIdx){
            weights[clIdx][centrIdx] = settings.initWeight;
         }
      }
   }
   
   double ConvSVM::output(vector< vector<double> >& activations, unsigned int svmIdx){
      unsigned int nCentroids = activations[0].size();
      double out = 0;
      
      for(size_t centrIdx = 0; centrIdx < nCentroids; ++centrIdx){
         out += weights[svmIdx][centrIdx] * activations[0][centrIdx];
      }
      return out;
   }
   
   void ConvSVM::train(vector< vector< vector<double> > >& activations, CSVMDataset* ds){
      
      unsigned int nData = activations.size();
      for(size_t svmIdx = 0; svmIdx < settings.nClasses; ++svmIdx){
         
         for(size_t itIdx; itIdx < settings.nIter; ++itIdx){
            double sumSlack = 0;
            for(size_t dIdx = 0; dIdx < nData; ++dIdx){
         
               unsigned int label = ds->getImagePtr(dIdx)->getLabelId();
               double yData = (label == svmIdx ? 1.0 : -1.0);
               double out = output(activations[dIdx], svmIdx);
               double dSlack = 0;
               double weightSum = 0;
               sumSlack += yData - out;
               //determine slack differential
               if(yData * out < 1){
                  for(size_t clIdx = 0; clIdx < settings.nCentroids; ++clIdx){
                     dSlack += activations[dIdx][0][clIdx];
                     weightSum += weights[svmIdx][clIdx];
                     
                  }
                  dSlack *= yData;
               }
               
               //update weights
               for(size_t clIdx = 0; clIdx < settings.nCentroids; ++clIdx){
                  weights[svmIdx][clIdx] -= settings.learningRate * (0.5 * weightSum + settings.CSVM_C * dSlack) ;
               }
               
               double objective = 0;
               for(size_t clIdx = 0; clIdx < settings.nCentroids; ++clIdx){
                  objective += weights[svmIdx][clIdx] * weights[svmIdx][clIdx];
               }
               objective += settings.CSVM_C * sumSlack;
               
               cout << "CSVM " << svmIdx << ": Objective = " << objective << endl;               
               
            }
         }
      }
   }
   
   unsigned int ConvSVM::classify(vector< vector<double> >& activations){
      unsigned int maxLabel = 0;
      double maxOut = numeric_limits<double>::min();
      for(size_t svmIdx = 1; svmIdx < settings.nClasses; ++svmIdx){
         double out = output(activations, svmIdx);
         if(out > maxOut){
            maxOut = out;
            maxLabel = svmIdx;
         }
      }
      return maxLabel;
      
   }
