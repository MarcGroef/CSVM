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

   }
   
   double ConvSVM::output(vector< vector<double> >& activations, unsigned int svmIdx){
      
      unsigned int nCentroids = activations[0].size();
      double out = 0;
      
      for(size_t centrIdx = 0; centrIdx < nCentroids; ++centrIdx){
         
         out += weights[svmIdx][centrIdx] * activations[0][centrIdx];
         
      }
      
      out += biases[svmIdx];
      return out;
   }
   
   void ConvSVM::train(vector< vector< vector<double> > >& activations, CSVMDataset* ds){
      
      unsigned int nData = activations.size();
  
      //for all csvms in ensemble
      for(size_t svmIdx = 0; svmIdx < settings.nClasses; ++svmIdx){
         
         for(size_t itIdx = 0; itIdx < settings.nIter; ++itIdx){
            
            double sumSlack = 0;

            for(size_t dIdx = 0; dIdx < nData; ++dIdx){
               
               unsigned int label = ds->getImagePtr(dIdx)->getLabelId();
               double yData = (label == svmIdx ? 1.0 : -1.0);
               double out = output(activations[dIdx], svmIdx);
               
               //add max(0, 1 - yData * out)
               sumSlack += 1 - yData * out < 0 ? 0 : 1 -  yData * out;
              
               for(size_t clIdx = 0; clIdx < settings.nCentroids; ++clIdx){
                  
                  if(yData * out < 1)
                     weights[svmIdx][clIdx] -= settings.learningRate *( (1.0/settings.CSVM_C )*weights[svmIdx][clIdx] -  yData * activations[dIdx][0][clIdx]) ;
                  else
                     weights[svmIdx][clIdx] -= settings.learningRate *( (1.0/settings.CSVM_C )*weights[svmIdx][clIdx]);
                  
               }
               
               if(yData * out < 1)
                  biases[svmIdx] += settings.learningRate * yData; 
                  
            }
            
            double objective = 0;
            for(size_t clIdx = 0; clIdx < settings.nCentroids; ++clIdx){
               objective += weights[svmIdx][clIdx] * weights[svmIdx][clIdx];
            }
            objective /= 2.0;
            objective += settings.CSVM_C * sumSlack;
            
            if(itIdx % 100 == 0)cout << "CSVM " << svmIdx << ": Objective = " << objective << ", sumSlack = " << sumSlack << endl;   
         }
         //biases[svmIdx] = tempBias ;
      }
   }
   
   
   //classify image, given its activations
   unsigned int ConvSVM::classify(vector< vector<double> >& activations){
      unsigned int maxLabel = 0;
      double maxOut = output(activations, 0);
      //cout << "out 0 = " << maxOut << endl;
      //cout << "settings.nClasses = " << settings.nClasses << endl;
      for(size_t svmIdx = 1; svmIdx < settings.nClasses; ++svmIdx){
         double out = output(activations, svmIdx);
         //cout << "out " << svmIdx << " = " << out << " with bias = " << biases[svmIdx] <<endl;
         if(out > maxOut){
            maxOut = out;
            maxLabel = svmIdx;
            //cout << "Maxlabel = " << maxLabel << endl;
         }
      }
      return maxLabel;
      
   }
