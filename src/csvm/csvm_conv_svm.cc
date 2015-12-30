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
      biases  = vector<double>(settings.nClasses, 0);
      maxOuts = vector<double>(settings.nClasses, 0);
      minOuts = vector<double>(settings.nClasses, 0);
      avOuts  = vector<double>(settings.nClasses, 0);
      //cout << "nCentroidsActs = " << activations[0].size();
      //for all csvms in ensemble
      ofstream statDatFile;

      for(size_t svmIdx = 0; svmIdx < settings.nClasses; ++svmIdx){
         stringstream ss;
         ss << "statData_SVM-" << svmIdx;
         string fName = ss.str();
         statDatFile.open ( fName.c_str() );
         statDatFile << "Iteration,Objective,Score,MinOut,MaxOut" << endl;
         cout << "\n\nSVM " << svmIdx << ":\t\t\t(data written to " << fName << ")\n" << endl;
         double prevObjective = numeric_limits<double>::max();
         double learningRate = settings.learningRate;

         for(size_t itIdx = 0; itIdx < settings.nIter; ++itIdx){
            
            double sumSlack = 0;
            float right = 0;
            float wrong = 0;

            minOut = 0;
            maxOut = 0;
            nMax   = 0;
            nMin   = 0;

            for(size_t dIdx = 0; dIdx < nData; ++dIdx){
               
               unsigned int label = ds->getTrainImagePtr(dIdx)->getLabelId();
               double yData = (label == svmIdx ? 1.0 : -1.0);
               double out = output(activations[dIdx], svmIdx);
               
               
              
               for(size_t centrIdx = 0; centrIdx < settings.nCentroids; ++centrIdx){

                  // original:
                  if(yData * out < 1)
                     weights[svmIdx][centrIdx] -= settings.learningRate * ( (weights[svmIdx][centrIdx] / settings.CSVM_C) -  yData * activations[dIdx][centrIdx]) ;
                  else
                     weights[svmIdx][centrIdx] -= settings.learningRate * ( (weights[svmIdx][centrIdx] / settings.CSVM_C) );

                  // my derivative:
                  // EFFECT: lower score on testset, comparable otherwise. Seems to perform better without the else statement.
                  //if(yData * out < 1)
                  //   weights[svmIdx][centrIdx] -= settings.learningRate * ( (weights[svmIdx][centrIdx] -  yData / settings.CSVM_C * activations[dIdx][centrIdx])) ;
                  //else
                  //   weights[svmIdx][centrIdx] -= settings.learningRate * ( weights[svmIdx][centrIdx] );

                  // L2
                  // EFFECT: Not obvious yet...
                  //if(yData * out < 1)
                  //   weights[svmIdx][centrIdx] -= settings.learningRate * (weights[svmIdx][centrIdx] + 1 / settings.CSVM_C * (-4 * yData * activations[dIdx][centrIdx]   +   2 * out * activations[dIdx][centrIdx])) ;
                  //else
                  //   weights[svmIdx][centrIdx] -= settings.learningRate * weights[svmIdx][centrIdx] ;
                                   
                  // L2 V2
                  // EFFECT: Not obvious yet...
                  //if(yData * out < 1)
                  //   weights[svmIdx][centrIdx] -= settings.learningRate * (weights[svmIdx][centrIdx] * 1 / settings.CSVM_C + (-4 * yData * activations[dIdx][centrIdx]   +   2 * out * activations[dIdx][centrIdx])) ;
                  //else
                  //   weights[svmIdx][centrIdx] -= settings.learningRate * weights[svmIdx][centrIdx]  * 1 / settings.CSVM_C;

                  // L2 VM1
                  // EFFECT: Works Nicely.. 45%
                  //if(yData * out < 1)
                  //   weights[svmIdx][centrIdx] -= -settings.learningRate * (weights[svmIdx][centrIdx] * 1 / settings.CSVM_C + ( (1-out*yData) * yData * activations[dIdx][centrIdx])) ;
                  //else
                  //   weights[svmIdx][centrIdx] -= -settings.learningRate * weights[svmIdx][centrIdx]  * 1 / settings.CSVM_C;

                  // L2 VM2
                  // EFFECT: Works Nicely.. 43%
                  //if(yData * out < 1)
                  //   weights[svmIdx][centrIdx] -= -settings.learningRate * (weights[svmIdx][centrIdx] * 1 / settings.CSVM_C + ( (1-out*yData) * yData * activations[dIdx][centrIdx])) ;
                  //else
                  //   weights[svmIdx][centrIdx] -= settings.learningRate * weights[svmIdx][centrIdx]  * 1 / settings.CSVM_C;

                  // L2 VM3
                  // EFFECT: Works Nicely.. 43%
                  //if(yData * out < 1){
                     //weights[svmIdx][centrIdx] += learningRate * (weights[svmIdx][centrIdx] * 1 / settings.CSVM_C + ( (1-out*yData) * yData * activations[dIdx][centrIdx])) ;
                     //++wrong;
                  //} else {
                     //weights[svmIdx][centrIdx] -= learningRate * weights[svmIdx][centrIdx] * 1 / settings.CSVM_C;
                     //++right;
                  //}
 
                  // L2 VM3
                  // EFFECT: Works Nicely.. 43%
                  if(yData * out < 1){
                     //weights[svmIdx][centrIdx] += learningRate * (weights[svmIdx][centrIdx] * 1 / settings.CSVM_C + ( (1-out*yData) * yData * activations[dIdx][centrIdx])) ;
                     weights[svmIdx][centrIdx] += learningRate * (weights[svmIdx][centrIdx] * 1 / settings.CSVM_C + ( 3*pow(1-out*yData, 2) * yData * activations[dIdx][centrIdx])) ;
                     ++wrong;
                  } else {
                     weights[svmIdx][centrIdx] -= learningRate * weights[svmIdx][centrIdx] * 1 / settings.CSVM_C;
                     ++right;
                  }

                  // L2 VM4
                  // EFFECT: Works Nicely.. 43%
                  //if(yData * out < 1 )// && rand()%10 > 2)
                  //   weights[svmIdx][centrIdx] += settings.learningRate * (weights[svmIdx][centrIdx] * 1 / settings.CSVM_C + ( (1-out*yData) * yData * activations[dIdx][centrIdx])) ;
               }//centrIdx
               
               //original bias function
               if(yData * out < 1)
                  biases[svmIdx] += settings.learningRate * yData;
               
               
               //if(yData * out < 1)
               //   biases[svmIdx] -= learningRate * (1-yData*out) * (-yData); 
               
               
               //add max(0, 1 - yData * out)               FOR L2 To Chi squared
               sumSlack += 1 - yData * out < 0 ? 0 : (1 -  yData * out) * (1 -  yData * out);
                  
               if (out > 0)  { maxOut += out; ++nMax; }
               if (out < 0)  { minOut += out; ++nMin; }

               //if(out > 0) maxOutSum += (isfinite(out) ? out : 0 );
               //if(out < 0) minOutSum += (isfinite(out) ? out : 0 );

               //if (!isfinite(maxOut)) maxOut = 0;
               //if (!isfinite(minOut)) minOut = 0;

            }//dIdx
            
            //measure objective 
            double objective = 0;

            for(size_t clIdx = 0; clIdx < settings.nCentroids; ++clIdx){
               objective += weights[svmIdx][clIdx] * weights[svmIdx][clIdx];
            }

            double hypPlane = objective;
            objective /= 2.0;
            objective += settings.CSVM_C * sumSlack ;
            
            maxOuts[svmIdx] = (double)  maxOut / nMax;
            minOuts[svmIdx] = (double)  minOut / nMin;
            avOuts[svmIdx]  = (double) (maxOut + minOut) / nData;

            if(itIdx % 100 == 0)cout << "CSVM " << svmIdx << ":\tObjective = " << objective << "\t Score = " << (right / (right+wrong) * 100.0) << endl;   
            statDatFile << itIdx << "," << objective << "," << float (right / (right+wrong) * 100) << "," << minOuts[svmIdx] << "," << maxOuts[svmIdx] << endl;

            if (objective > prevObjective) learningRate *= 0.999;
            prevObjective = objective;

         }//itIdx
         
        statDatFile.close();
         
      }//svmIdx
   }
   
   
   //classify image, given its activations
   unsigned int ConvSVM::classify(vector<double>& activations){
      unsigned int maxLabel = 0;
      double maxVal = output(activations, 0);
      cout << "\n";
      for(size_t svmIdx = 0; svmIdx < settings.nClasses; ++svmIdx){
         double out = output(activations, svmIdx);
         cout << "out " << svmIdx << " = " << out << "  /  " << minOuts[svmIdx] << " - "  << maxOuts[svmIdx] << "    Bias = " << biases[svmIdx] <<endl;
         if(out > maxVal){
            maxVal = out;
            maxLabel = svmIdx;
            //cout << "Maxlabel = " << maxLabel << endl;
         }
      }
      return maxLabel;
      
   }
