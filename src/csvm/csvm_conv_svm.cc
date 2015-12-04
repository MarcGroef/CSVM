#include<csvm/csvm_conv_svm.h>

using namespace std;
using namespace csvm;

   void ConvSVM::setSettings(ConvSVMSettings s){
      settings = s;
      
      cout << "nCentroids = " << settings.nCentroids << endl;
      cout << "nClasses = " << settings.nClasses << endl;
      cout << "initWeight= " << settings.initWeight << endl;
      weights = vector< vector<double> >(settings.nClasses, vector<double>(settings.nCentroids, settings.initWeight));
      biases = vector<double>(settings.nClasses, 0);
      /*for(size_t clIdx = 0; clIdx < settings.nClasses; ++clIdx){
         weights[clIdx].reserve(settings.nCentroids);
         for(size_t centrIdx = 0; centrIdx < settings.nCentroids; ++centrIdx){
            weights[clIdx][centrIdx] = settings.initWeight;
         }
      }*/
   }
   
   double ConvSVM::output(vector< vector<double> >& activations, unsigned int svmIdx){
      unsigned int nCentroids = activations[0].size();
      double out = 0;
      
      for(size_t centrIdx = 0; centrIdx < settings.nCentroids; ++centrIdx){
         //cout << "weights = " << weights[svmIdx][centrIdx] << endl;
         out += weights[svmIdx][centrIdx] * activations[0][centrIdx];
      }
      out += biases[svmIdx];
      return out;
   }
   
   void ConvSVM::train(vector< vector< vector<double> > >& activations, CSVMDataset* ds){
      
      unsigned int nData = activations.size();
      for(size_t svmIdx = 0; svmIdx < settings.nClasses; ++svmIdx){
         
         for(size_t itIdx = 0; itIdx < settings.nIter; ++itIdx){
            double sumSlack = 0;
            
            
            
            //cout << "sumdSlack = " << sumDSlack << endl;
            for(size_t dIdx = 0; dIdx < nData; ++dIdx){
               double sumDSlack = 0;
               
               
               for(size_t dIdx1 = 0; dIdx1 < nData; ++dIdx1){
                  unsigned int label = ds->getImagePtr(dIdx1)->getLabelId();
                  double yData = (label == svmIdx ? 1.0 : -1.0);
                  double updatesdSlack = 0;
                  double out = output(activations[dIdx1], svmIdx);
                  if(yData * out < 1){
                     for(size_t clIdx = 0; clIdx < settings.nCentroids; ++clIdx){
                        updatesdSlack += activations[dIdx1][0][clIdx];
                     }
                  }
                  updatesdSlack *= yData;
                  sumDSlack += updatesdSlack;
               }
               
               
               unsigned int label = ds->getImagePtr(dIdx)->getLabelId();
               double yData = (label == svmIdx ? 1.0 : -1.0);
               double out = output(activations[dIdx], svmIdx);
               
               double weightSum = 0;
               sumSlack += yData * out < 0 ? yData * out * -1 : yData * out;
               //determine slack differential
               if(yData * out < 1){  //otherwise keep it zero
                  for(size_t clIdx = 0; clIdx < settings.nCentroids; ++clIdx){
                     //dSlack += activations[dIdx][0][clIdx];
                     weightSum += weights[svmIdx][clIdx] * weights[svmIdx][clIdx];
                     
                     
                  }
                  //dSlack *= yData;
               }
               
               //update weights
               for(size_t clIdx = 0; clIdx < settings.nCentroids; ++clIdx){
                  //cout << "weight = " << weights[svmIdx][clIdx] << ", weightSum = " << weights[svmIdx][clIdx] << "sumDSlack = " << sumDSlack << endl;
                  weights[svmIdx][clIdx] -= settings.learningRate * ( weights[svmIdx][clIdx] + settings.CSVM_C * sumDSlack) ;
                  
               }
               
                           
               //biases[svmIdx] += settings.learningRate *  yData; 
            }
            double objective = 0;
            for(size_t clIdx = 0; clIdx < settings.nCentroids; ++clIdx){
               objective += weights[svmIdx][clIdx] * weights[svmIdx][clIdx];
            }
            objective /= 2.0;
            objective += settings.CSVM_C * sumSlack;
            
            /*if(itIdx % 100 == 0)*/cout << "CSVM " << svmIdx << ": Objective = " << objective << ", sumSlack = " << sumSlack << endl;   
         }
      }
   }
   
   unsigned int ConvSVM::classify(vector< vector<double> >& activations){
      unsigned int maxLabel = 0;
      double maxOut = output(activations, 0);
      cout << "out 0 = " << maxOut << endl;
      for(size_t svmIdx = 1; svmIdx < settings.nClasses; ++svmIdx){
         double out = output(activations, svmIdx);
         cout << "out " << svmIdx << " = " << out << endl;
         if(out > maxOut){
            maxOut = out;
            maxLabel = svmIdx;
         }
      }
      return maxLabel;
      
   }
