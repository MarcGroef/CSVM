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
   
  
   void ConvSVM::train(vector< vector< Feature > > dataFeaturesVec, CSVMDataset* ds, Codebook cb){
      
      settings.nCentroids = cb.getNCentroids();

      unsigned int nData = dataFeaturesVec.size();
      vector < double > activations;
      weights = vector< vector<double> >(settings.nClasses, vector<double>(settings.nCentroids, settings.initWeight));
      biases  = vector<double>(settings.nClasses, 0);
      maxOuts = vector<double>(settings.nClasses, 0);
      minOuts = vector<double>(settings.nClasses, 0);
      avOuts  = vector<double>(settings.nClasses, 0);
      double decay = 2;

      //for all csvms in ensemble
      ofstream statDatFile;
      cout << scientific << setprecision(5);

      for(size_t svmIdx = 0; svmIdx < settings.nClasses; ++svmIdx){
         stringstream ss;
         ss << "statData_SVM-" << svmIdx << ".csv";
         string fName = ss.str();
         statDatFile.open ( fName.c_str() );
         statDatFile << "Iteration,Objective,Score,MinOut,MaxOut,stdDevMinOutPos,stdDevMinOutNeg,StdDevMaxOutPos,StdDevMaxOutNeg,HyperplanePercentage" << endl;
         cout << "\n\nSVM " << svmIdx << ":\t\t\t(data written to " << fName << ")\n" << endl;
         double prevObjective = numeric_limits<double>::max();
         double learningRate = settings.learningRate;
int switchVar = 3;
         for(size_t itIdx = 0; itIdx < settings.nIter; ++itIdx){
            --switchVar;
            double sumSlack = 0;
            float right = 0;
            float wrong = 0;

            minOut = 0;
            maxOut = 0;
            nMax   = 0;
            nMin   = 0;

            allOuts  = vector<double>(nData, 0);

            for(size_t dIdx = 0; dIdx < nData; ++dIdx){
               
               activations = cb.getQActivationsBackProp(dataFeaturesVec[dIdx], svmIdx);
               unsigned int label = ds->getTrainImagePtr(dIdx)->getLabelId();
               double yData = (label == svmIdx ? 1.0 : -1.0);
               double out = output(activations, svmIdx);
              
//cout << "0" << endl;
               for(size_t centrIdx = 0; centrIdx < settings.nCentroids; ++centrIdx){
//cout << "\n\n\n" << centrIdx << endl;
//cout << "1" << endl;
                  // original:
             //     if(yData * out < 1){
             //        weights[svmIdx][centrIdx] -= settings.learningRate * ( (weights[svmIdx][centrIdx] / settings.CSVM_C) -  yData * activations[dIdx][centrIdx]) ;
             //        ++wrong;
             //     }else{
             //        weights[svmIdx][centrIdx] -= settings.learningRate * ( (weights[svmIdx][centrIdx] / settings.CSVM_C) );
             //        ++right;
             //     }


                  // L2 VM3
                  // EFFECT: Graphs seem better... Accuracy does not increase enormously
                  if(yData * out < 1){
                     weights[svmIdx][centrIdx] += learningRate * (weights[svmIdx][centrIdx] * 1 / settings.CSVM_C + ( (1-out*yData) * yData * activations[centrIdx])) ;
              //       weights[svmIdx][centrIdx] += learningRate * activations[dIdx][centrIdx] * (weights[svmIdx][centrIdx] * 1 / settings.CSVM_C + ( 3*pow(1-out*yData, 2) * yData * activations[dIdx][centrIdx])) ;
                     ++wrong;
//cout << "2" << endl;
                  } else {
                     weights[svmIdx][centrIdx] += learningRate * weights[svmIdx][centrIdx] * 1 / settings.CSVM_C ;
                     ++right;
//cout << "3" << endl;
                  }

               }//centrIdx

//if (itIdx > 10)
               if (yData * out < 1){
                  cb.applyBackProp(weights[svmIdx], yData, learningRate, out, svmIdx);
                  switchVar = 3;
               }
//cout << "4" << endl;
               //bias function
               if(yData * out < 1)
                  biases[svmIdx] += settings.learningRate * (yData - out);
               
//cout << "5" << endl;
               //add max(0, 1 - yData * out)               FOR L2 To Chi squared
               sumSlack += 1 - yData * out < 0 ? 0 : (1 -  yData * out) * (1 -  yData * out);
                  
//cout << "6" << endl;
               if (out > 0)  { maxOut += out; ++nMax; }
               if (out < 0)  { minOut += out; ++nMin; }
               allOuts[dIdx] = out;

//cout << "7" << endl;
            }//dIdx
            
//cout << "0a" << endl;
            //measure objective 
            double objective = 0;

            for(size_t clIdx = 0; clIdx < settings.nCentroids; ++clIdx){
               objective += weights[svmIdx][clIdx] * weights[svmIdx][clIdx];
            }

//cout << "1a" << endl;
            double hypPlane = objective;
            objective /= 2.0;
            objective += settings.CSVM_C * sumSlack ;
            
            maxOuts[svmIdx] = (double)  maxOut / nMax;
            minOuts[svmIdx] = (double)  minOut / nMin;
            avOuts[svmIdx]  = (double) (maxOut + minOut) / nData;

//cout << "2a" << endl;
            double stdDevMaxOutPos = 0;
            double stdDevMaxOutNeg = 0;
            double stdDevMinOutPos = 0;
            double stdDevMinOutNeg = 0;
            int nMaxPos = 0;
            int nMaxNeg = 0;
            int nMinPos = 0;
            int nMinNeg = 0;

//cout << "3a" << endl;
            for (size_t dIdx=0; dIdx < nData; ++dIdx){
               if (allOuts[dIdx] >  0)   {
                  if (allOuts[dIdx] >  maxOuts[svmIdx]) {stdDevMaxOutPos += pow((allOuts[dIdx] - maxOuts[svmIdx]), 2); ++nMaxPos;}
                  if (allOuts[dIdx] <= maxOuts[svmIdx]) {stdDevMaxOutNeg += pow((allOuts[dIdx] - maxOuts[svmIdx]), 2); ++nMaxNeg;}
               }   
               if (allOuts[dIdx] <= 0)   {
                  if (allOuts[dIdx] >  minOuts[svmIdx]) {stdDevMinOutPos += pow((allOuts[dIdx] - minOuts[svmIdx]), 2); ++nMinPos;}
                  if (allOuts[dIdx] <= minOuts[svmIdx]) {stdDevMinOutNeg += pow((allOuts[dIdx] - minOuts[svmIdx]), 2); ++nMinNeg;}
               }   
            }
            
//cout << "4a" << endl;
            stdDevMaxOutPos = sqrt(stdDevMaxOutPos / nMaxPos);
            stdDevMaxOutNeg = sqrt(stdDevMaxOutNeg / nMaxNeg);
            stdDevMinOutPos = sqrt(stdDevMinOutPos / nMinPos);
            stdDevMinOutNeg = sqrt(stdDevMinOutNeg / nMinNeg);
	

//cout << "5a" << endl;
            if(itIdx % 1 == 0)cout << "CSVM " << svmIdx << " (" << itIdx << ")" << ":\tObjective = " << objective << "\t Score = " << fixed << (right / (right+wrong) * 100.0) << scientific << "\tBias: " << biases[svmIdx] << endl;   
            statDatFile << itIdx << "," << objective << "," << fixed << float (right / (right+wrong) * 100) << "," << scientific << minOuts[svmIdx] << "," << maxOuts[svmIdx] << "," << stdDevMinOutPos << "," << stdDevMinOutNeg << "," << stdDevMaxOutPos << "," << stdDevMaxOutNeg << "," << hypPlane / objective * 100 << endl;

            if (objective > prevObjective) learningRate *= 0.75;
            prevObjective = objective;

//cout << "6a" << endl;
         }//itIdx
         
//cout << "7a" << endl;
         statDatFile.close();

         ss << "W-Data_SVM-" << svmIdx << ".csv";
         fName = ss.str();
         statDatFile.open ( fName.c_str() );
         statDatFile << "Dimension,Value" << endl;

         for (size_t cIdx=0; cIdx < settings.nCentroids; ++cIdx)
            statDatFile << cIdx << "," << weights[svmIdx][cIdx] << endl;

         statDatFile.close();
//cout << "8a" << endl;
      }//svmIdx
   }
   
   
   //classify image, given its activations
   unsigned int ConvSVM::classify(vector < vector<double> >& activations){
      unsigned int maxLabel = 0;
      //double maxVal = (output(activations, 0) > 0) ? output(activations, 0) / maxOuts[0] : output(activations, 0) / -minOuts[0];
      double maxVal = output(activations[0], 0);
      //cout << "\n";
      for(size_t svmIdx = 1; svmIdx < settings.nClasses; ++svmIdx){
         //double out = (output(activations, svmIdx) > 0) ? output(activations, svmIdx) / maxOuts[svmIdx] : output(activations, svmIdx) / -minOuts[svmIdx];
         double out = output(activations[svmIdx], svmIdx);
         //cout << "out " << svmIdx << " = " << out << "\t  /  " << minOuts[svmIdx] << " - "  << maxOuts[svmIdx] << "    Bias = " << biases[svmIdx] <<endl;
         if(out > maxVal){
            maxVal = out;
            maxLabel = svmIdx;
            //cout << "Maxlabel = " << maxLabel << endl;
         }
      }
      return maxLabel;
      
   }
