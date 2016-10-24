#include <csvm/csvm_svm.h>
#include <algorithm>
/*This class is a class initially built to have a basic working pipeline.
 * The SVM class contains a L2-SVM. Sometimes things are called 'classic', since we 'experimented' a bit with the objective function ed.
 * 
 */

using namespace std;
using namespace csvm;

/*
 * SVM Constructor: It initializes the SVM.
 * */
SVM::SVM(int datasetSize, int nClasses, int nCentroids, unsigned int labelId){
	
   this->nCentroids = nCentroids;
   this->datasetSize = datasetSize;
   this->nClasses = nClasses;
   this->classId = labelId;  //target label

   //initialize the bias to zero
   bias = 0;
}

/*
 * Aquire parameters from the settings file
 * */
void SVM::setSettings(SVM_Settings s){
   //cout << "I recieved " << s.alphaDataInit << endl;
   settings = s;
   alphaData = vector<float>(datasetSize,settings.alphaDataInit /** settings.SVM_C_Data*/);
}



//update alphaData for dual obj. SVM with alpha_i, alpha_j, given a similarity kernel between two activation vectors:
float SVM::updateAlphaDataClassic(vector< vector< float >  >& simKernel, CSVMDataset* ds){
   
   float diff = 0.0;
   float deltaDiff = 0.0;
   float target;
   float sum = 0.0;
   float yData0;
   float yData1;
   unsigned int nData = simKernel.size();
   float deltaAlpha;
   unsigned int kernelIdx0,kernelIdx1;
	
	//alpha_i is called alphaData[dIdx0] here, where alpha_j is called alphaData[dIdx1];
	
   //for all alpha's:
   for(size_t dIdx0 = 0; dIdx0 < nData; ++dIdx0){
		
      deltaDiff = 0.0;
      yData0 = ((ds->getTrainImagePtr(dIdx0)->getLabelId()) == classId ? 1.0 : -1.0);
      
		
      //calculate the sum
		sum = 0.0;
      for(size_t dIdx1 = 0; dIdx1 < nData; ++dIdx1){
         yData1 = ((ds->getTrainImagePtr(dIdx1)->getLabelId()) == classId ? 1.0 : -1.0);
         if(dIdx1 > dIdx0){
            kernelIdx0 = dIdx1;
            kernelIdx1 = dIdx0;
         }else{
            kernelIdx0 = dIdx0;
            kernelIdx1 = dIdx1;
         }
         
         sum += alphaData[dIdx1] * yData0 * yData1 * simKernel[kernelIdx0][kernelIdx1];
         //cout << "alphaD = " << alphaData[dIdx1] << " Yi = " << yData0 << " Yj = " << yData1 <<" simkern = " << simKernel[dIdx0][dIdx1] << endl;
         
      }
            
      deltaAlpha = 1.0 - sum;
      //calc new value
      target = settings.D2 * alphaData[dIdx0] + deltaAlpha * settings.learningRate;
      //keep it in boundaries
      
      target = target > settings.SVM_C_Data ? settings.SVM_C_Data : target;
      target = target < 0.0 ? 0.0 : target;
      /*if(settings.kernelType != LINEAR)*/ 
      //else target = target < settings.SVM_C_Data ? settings.SVM_C_Data : target;
      deltaDiff = alphaData[dIdx0] - target;
      diff += (deltaDiff < 0.0 ? deltaDiff * -1.0 : deltaDiff);
      //set new value
      alphaData[dIdx0] = target;
      
   }
   
   return diff;
}

//make sure  sum(alphaCentroid * yCentroid) == 0, or below threshold
float SVM::constrainAlphaDataClassic(vector< vector<float> >& simKernel, CSVMDataset* ds){
   
   
   float oldVal;
   unsigned int nData = simKernel.size();
   float deltaAlpha;
   float yData; 
   float target;
   
   float diff = 0.0;
   float deltaDiff = 0.0;
   float sum = 0;
   float threshold = 0.00001;
   
   //calculate current sum
   for(size_t dIdx0 = 0; dIdx0  < nData; ++dIdx0){
         yData = (classId == (unsigned int)(ds->getTrainImagePtr(dIdx0)->getLabelId()) ? 1.0 : -1.0);
         sum += alphaData[dIdx0] * yData;
   }
   
   //while sum is above threshold, update alphas
   for(size_t constItr = 0; sum > threshold /*|| sum < 0-threshold*/; ++constItr){
      for(size_t dIdx0 = 0; dIdx0  < nData; ++dIdx0){
         
         yData = (classId == (unsigned int)(ds->getTrainImagePtr(dIdx0)->getLabelId()) ? 1.0 : -1.0);
         oldVal = alphaData[dIdx0];
         deltaAlpha = -2.0 * settings.cost * sum * yData;
         
         //make sure that 0 <= alphaCentroid <= SVM_C_Centroid
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

void SVM::calculateBiasClassic(vector< vector< float> >& simKernel, CSVMDataset* ds){
   bias = 0;
   unsigned int total = 0;
   unsigned int nData = simKernel.size();
   float output;
   float yData;
   unsigned int kernelIdx0, kernelIdx1;
   
   for(size_t dIdx0 = 0; dIdx0 < nData; ++dIdx0){
     
      if((alphaData[dIdx0] > 0) && (alphaData[dIdx0] < settings.SVM_C_Data - 0.000001)){
         output = 0;
         for(size_t dIdx1 = 0; dIdx1 < nData; ++dIdx1){
            if(dIdx1 > dIdx0){
               kernelIdx0 = dIdx1;
               kernelIdx1 = dIdx0;
            }else{
               kernelIdx0 = dIdx0;
               kernelIdx1 = dIdx1;
            }
            
            
            yData = ((unsigned int)(ds->getTrainImagePtr(dIdx1)->getLabelId()) == classId ? 1.00 : -1.00);
            output += alphaData[dIdx1] * yData * simKernel[kernelIdx0][kernelIdx1];
            
         }
         yData = ((unsigned int)(ds->getTrainImagePtr(dIdx0)->getLabelId()) == classId ? 1.00 : -1.00);
         bias += yData - output;
         ++total;
      }
   }
   
   if(total == 0)
      bias = 0;
   else 
      bias /= total;
}
void SVM::trainClassic(vector< vector< float> >& simKernel, CSVMDataset* ds){
   
   float sumDeltaAlpha = 1000.0;
   //float prevSumDeltaAlpha = 100.0;
   float deltaAlphaData;


   //float objective = 0.0;
   float sum0 = 0.0;
   float sum1 = 0.0;
   
   unsigned int kernelIdx0, kernelIdx1;
   for(size_t round = 0; /*sumDeltaAlpha > 0.00001*/ /*(prevObjective - objective < -0.0001 || round < 1000)*/ round < settings.nIterations; ++round){

      //prevSumDeltaAlpha = sumDeltaAlpha;
      sumDeltaAlpha = 0.0;
      deltaAlphaData = updateAlphaDataClassic(simKernel, ds);
      deltaAlphaData = deltaAlphaData < 0.0 ? deltaAlphaData * -1.0 : deltaAlphaData;
      sumDeltaAlpha += deltaAlphaData;
      
      constrainAlphaDataClassic(simKernel, ds);
      
      
      float yData0, yData1;
      
      //calculate objective
      sum0 = 0.0;
      sum1 = 0.0;
      for(size_t dIdx0 = 0; dIdx0 < simKernel.size(); ++dIdx0){
         yData0 = (ds->getTrainImagePtr(dIdx0)->getLabelId() == classId ? 1.0 : -1.0);
         
         sum0 += alphaData[dIdx0];
         for(size_t dIdx1 = 0; dIdx1 < simKernel.size(); ++dIdx1){
            if(dIdx1 > dIdx0){
               kernelIdx0 = dIdx1;
               kernelIdx1 = dIdx0;
            }else{
               kernelIdx0 = dIdx0;
               kernelIdx1 = dIdx1;
            }
            
            
            yData1 = (ds->getTrainImagePtr(dIdx1)->getLabelId() == classId ? 1.0 : -1.0);
            
            sum1 += alphaData[dIdx0] * alphaData[dIdx1] * yData0 *  yData1 * simKernel[kernelIdx0][kernelIdx1];
            
         }
      }
      //objective = sum0 - 0.5 * sum1;
         
     //if(round % 100 == 0 )cout << "SVM " << classId << " training round " << round << ".  Sum of Change  = " << fixed << sumDeltaAlpha << "\tDeltaSOC = " << (prevSumDeltaAlpha - sumDeltaAlpha) << "  \tObjective : " << objective << endl;   
      //compute trainings-score:
      
   }
   calculateBiasClassic(simKernel, ds);
   //for(size_t aIdx = 0; aIdx < alphaData.size(); ++aIdx)
     //    cout << "Alpha " << aIdx << " = " << alphaData[aIdx] << endl;
}


//Classify an image, represented by a set of features, using the classic (alpha_i, alpha_j) SVM
float SVM::classifyClassic(vector< float >f, vector< vector<float> >& datasetActivations, CSVMDataset* ds){
   
   float result = 0;
   unsigned int nData = datasetActivations.size();
   float yData;
   float kernel = 0.0;
   unsigned int nCentroids = datasetActivations[0].size();
   Feature dataKernel(nData,0.0);
   

   
   //update sum with similarity between image activations
   for(size_t dIdx0 = 0; dIdx0 < nData; ++dIdx0){
      
      float sum = 0;
      //calculate similarity between activation vectors
      
      
      //RBF kernel:
      if(settings.kernelType == RBF){
         
         for(size_t centr = 0; centr < nCentroids; ++centr){
            //sum of distance squared
            sum += (datasetActivations[dIdx0][centr] - f[centr])*(datasetActivations[dIdx0][centr] - f[centr]);
         }
         
         //calculate kernel value:
         kernel = exp((-1.0* sum)/settings.sigmaClassicSimilarity);
         //cout << "Test kernel value : " << kernel << endl;
      }else if (settings.kernelType == LINEAR){
     
         //Linear kernel
         for(size_t centr = 0; centr < nCentroids; ++centr){
            sum += (datasetActivations[dIdx0][centr] * f[centr]);
         }
         
         kernel = sum;
         
      }else{
         cout << "CSVM::svm::Error! No valid kernel type selected! Try: RBF or LINEAR\n"  ;
      }
      
      
      
     yData = (ds->getTrainImagePtr(dIdx0)->getLabelId()) == classId ? 1.0 : -1.0;
     //add the result for this alpha
     //cout << yData << endl;
     result += alphaData[dIdx0] * yData * kernel;
      
   }
   //add bias to result
   result += bias;
   //cout << "SVM " << classId << " has bias " << bias << endl;
   //cout << "SVM " << classId << " says " << result << endl;
   return result;
}
