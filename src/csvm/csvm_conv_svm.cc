#include<csvm/csvm_conv_svm.h>

using namespace std;
using namespace csvm;

   void ConvSVM::setSettings(ConvSVMSettings s){
      settings = s;
      
      //cout << "nCentroids = " << settings.nCentroids << endl;
      //cout << "nClasses = " << settings.nClasses << endl;
      //cout << "initWeight= " << settings.initWeight << endl;
      


   }
   
   double ConvSVM::output(vector<double>& activations, unsigned int svmIdx){
      
      unsigned int nCentroids = activations.size();
      double out = 0;
      
      for(size_t centrIdx = 0; centrIdx < nCentroids; ++centrIdx){
         
         out += weights[svmIdx][centrIdx] * activations[centrIdx];
         
      }
      
      out += biases[svmIdx];
      return out;
   }
   
   void ConvSVM::train(vector< vector<double> >& activations, CSVMDataset* ds){
      
      unsigned int nData = activations.size();
      settings.nCentroids = activations[0].size();
      
      weights = vector< vector<double> >(settings.nClasses, vector<double>(settings.nCentroids, settings.initWeight));
      biases = vector<double>(settings.nClasses, 0);
      //cout << "nCentroidsActs = " << activations[0].size();
      //for all csvms in ensemble
      ofstream statDatFile;

      for(size_t svmIdx = 0; svmIdx < settings.nClasses; ++svmIdx){
         stringstream ss;
         ss << "statData_SVM-" << svmIdx;
         string fName = ss.str();
         statDatFile.open ( fName.c_str() );
         statDatFile << "itIdx\tObjective\t|HyperPlane|" << endl;
         cout << "\n\nSVM " << svmIdx << ":\t\t\t(data written to " << fName << ")\n" << endl;

         for(size_t itIdx = 0; itIdx < settings.nIter; ++itIdx){
            
            double sumSlack = 0;

            for(size_t dIdx = 0; dIdx < nData; ++dIdx){
               
               unsigned int label = ds->getTrainImagePtr(dIdx)->getLabelId();
               double yData = (label == svmIdx ? 1.0 : -1.0);
               double out = output(activations[dIdx], svmIdx);
               
               
              
               for(size_t centrIdx = 0; centrIdx < settings.nCentroids; ++centrIdx){

                  // original:
                  //if(yData * out < 1)
                  //   weights[svmIdx][centrIdx] -= settings.learningRate * ( (weights[svmIdx][centrIdx] / settings.CSVM_C) -  yData * activations[dIdx][centrIdx]) ;
                  //else
                  //   weights[svmIdx][centrIdx] -= settings.learningRate * ( (weights[svmIdx][centrIdx] / settings.CSVM_C) );

                  // my derivative:
                  // EFFECT: lower score on testset, comparable otherwise. Seems to perform better without the else statement.
                  //if(yData * out < 1)
                  //   weights[svmIdx][centrIdx] -= settings.learningRate * ( (weights[svmIdx][centrIdx] -  yData / settings.CSVM_C * activations[dIdx][centrIdx])) ;
                  //else
                  //   weights[svmIdx][centrIdx] -= settings.learningRate * ( weights[svmIdx][centrIdx] );

                  // L2
                  // EFFECT: Not obvious yet...
                  if(yData * out < 1)
                     weights[svmIdx][centrIdx] -= settings.learningRate * (weights[svmIdx][centrIdx] +  1 / settings.CSVM_C * (-4 * yData * activations[dIdx][centrIdx]   +   2 * out * activations[dIdx][centrIdx])) ;
                  else
                     weights[svmIdx][centrIdx] -= settings.learningRate * weights[svmIdx][centrIdx] ;
                                   
               }
               
               if(yData * out < 1)
                  biases[svmIdx] += settings.learningRate * yData; 
               
               
               //add max(0, 1 - yData * out)
               sumSlack += 1 - yData * out < 0 ? 0 : 1 -  yData * out;
                  
            }
            
            //measure objective 
            double objective = 0;
            for(size_t clIdx = 0; clIdx < settings.nCentroids; ++clIdx){
               
               objective += weights[svmIdx][clIdx] * weights[svmIdx][clIdx];
            }
            double hypPlane = objective;
            objective /= 2.0;
            objective += 1 / settings.CSVM_C * sumSlack;
            
            if(itIdx % 100 == 0)cout << "CSVM " << svmIdx << ":\tObjective = " << objective << "\t |HyperPlane| = " << hypPlane << endl;   
            statDatFile << itIdx << "\t" << objective << "\t" << hypPlane << endl;
         }
         statDatFile.close();
         
      }
   }
   
   
   //classify image, given its activations
   unsigned int ConvSVM::classify(vector<double>& activations){
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
