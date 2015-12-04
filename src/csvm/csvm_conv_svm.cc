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
         for(size_t dIdx = 0; dIdx < nData; ++dIdx){
            unsigned int label = ds->getImagePtr(dIdx)->getlabelId();
            double yData = (label == svmIdx ? 1.0 : -1.0);
            double out = output(activations[dIdx], svmIdx);
            double dSlack = (yData * out >= 1 ? 0 : yData * activations[dIdx][0])
            
         }
      ]
   }
   
   unsigned int ConvSVM::classify(vector< vector<double> >& activations){
      
      
   }
