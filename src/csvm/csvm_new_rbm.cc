#include <csvm/csvm_new_rbm.h>

using namespace std;
using namespace csvm;


void normalize(vector<double>& x){
   size_t vecSize = x.size();
   double sum = 0;
   
   for(size_t xIdx = 0; xIdx != vecSize; ++xIdx){
      sum += x[xIdx] < 0 ? x[xIdx] * -1.0 : x[xIdx];
   }
   for(size_t xIdx = 0; xIdx != vecSize; ++xIdx){
      x[xIdx] /= sum;
   }
}

void nrbm_standardize(vector<double>& x){
   
   size_t vecSize = x.size();
   double sum = 0;
   
   for(size_t xIdx = 0; xIdx != vecSize; ++xIdx){
      sum += x[xIdx];
   }
   double mean = sum / vecSize;
   double stddev = 0;
   
   for(size_t xIdx = 0; xIdx != vecSize; ++xIdx){
      stddev += (x[xIdx] - mean) * (x[xIdx] - mean);
   }
   stddev = sqrt(stddev + 0.001);
   stddev /= vecSize;
   
   for(size_t xIdx = 0; xIdx != vecSize; ++xIdx){
      x[xIdx] = (x[xIdx] - mean) / stddev;
   }
   
}

void maxDivision(vector<double>& x){
   size_t vecSize = x.size();
   double max = 0;
   
   for(size_t xIdx = 0; xIdx != vecSize; ++xIdx){
      if (x[xIdx] > max)
         max = x[xIdx];
   }
   
   
   for(size_t xIdx = 0; xIdx != vecSize; ++xIdx){
      x[xIdx] /= max;
   }
}

void NRBM::flowUp(){
   size_t inSize = settings.inputSize;
   size_t outSize = settings.outputSize;
   
   for(size_t outIdx = 0; outIdx != outSize; ++outIdx){
      double outSum = 0;
      for(size_t inIdx = 0; inIdx != inSize; ++inIdx){
         outSum += inputLayer[inIdx] * weights[inIdx][outIdx];
      }
      outputLayer[outIdx] = sigmoid(outSum + biasUp[outIdx]);
   }
}

void NRBM::flowDown(){
   size_t inSize = settings.inputSize;
   size_t outSize = settings.outputSize;
   
   for(size_t inIdx = 0; inIdx != inSize; ++inIdx){
      double inSum = 0;
      for(size_t outIdx = 0; outIdx != outSize; ++outIdx){
         inSum += outputLayer[outIdx] * weights[inIdx][outIdx];
      }
      inputLayer[inIdx] = sigmoid(inSum + biasDown[inIdx]);
   }
}

double NRBM::sigmoid(double x){
  
   return 1.0/ (1.0 + exp(-1.0 * x));
}

void NRBM::calculateEnergy(vector< vector<double> >& e){
   size_t inSize = settings.inputSize;
   size_t outSize = settings.outputSize;
   
   for(size_t inIdx = 0; inIdx != inSize; ++inIdx){
      for(size_t outIdx = 0; outIdx != outSize; ++outIdx){
         e[inIdx][outIdx] = inputLayer[inIdx] * outputLayer[outIdx];
      }
   }
}


double NRBM::studyFeature(vector<double>& f){
   size_t inSize = settings.inputSize;
   size_t outSize = settings.outputSize;
   
   
   //set data to input layer
   inputLayer = f;
   maxDivision(inputLayer);
   //normalize(inputLayer);
   //nrbm_standardize(inputLayer);
   
   
   //normalize vector
   //normalize(inputLayer);
   size_t nGibbs = settings.nGibbsSteps;
   
   vector<double> outData, inData;
   
   
   //get outputLayer activations
   flowUp();
   
   //store energy and layer info, before gibbs sampling
   calculateEnergy(dataEnergy);
   outData = outputLayer;
   inData = inputLayer;
   
   for(size_t gibbIdx = 0; gibbIdx != nGibbs; ++gibbIdx){
      flowDown();
      flowUp();
   }
   calculateEnergy(modelEnergy);
   //learning nGibbsSteps
   double deltaSum = 0;
   for(size_t inIdx = 0; inIdx != inSize; ++inIdx){
      for(size_t outIdx = 0; outIdx != outSize; ++outIdx){
         //update weights
         
         double update = settings.learningRate * (dataEnergy[inIdx][outIdx] - modelEnergy[inIdx][outIdx]);
         weights[inIdx][outIdx] += update;
         deltaSum += update < 0 ? -1.0 * update : update ;
         
         //update biases
         update = settings.learningRate * (outData[outIdx] - outputLayer[outIdx]);
         biasUp[outIdx] += update;
         
         update = settings.learningRate * (inData[inIdx] - inputLayer[inIdx]) ;
         biasDown[inIdx] += update;
         
      }
   }
   return deltaSum;
}

void NRBM::learn(vector< vector<double> >& data){
   unsigned int nIter = settings.nIterations;
   unsigned int nData = data.size();
   double delta;
   //for all iterations
   for(size_t itIdx = 0; itIdx != nIter; ++itIdx){
      //for all data
      for(size_t dataIdx = 0; dataIdx != nData; ++dataIdx){
         delta = studyFeature(data[dataIdx]);
      }
      //give message so now and then
      if(itIdx % 10 == 0){
         cout << "RBM iter " << itIdx << " delta = " << delta << endl;
      }
   }
}

vector<double> NRBM::describe(vector<double>& f){
   inputLayer = f;
   
   //nrbm_standardize(inputLayer);
   //normalize(inputLayer);
   maxDivision(inputLayer);
   flowUp();
   size_t nDims = settings.inputSize;
   /*cout << "in:\n";   
   for(size_t dIdx = 0; dIdx != nDims; ++dIdx){
      cout << inputLayer[dIdx] << ", ";
   }*/
   nDims = settings.outputSize;
   
   cout << "\nout:\n";
   for(size_t dIdx = 0; dIdx != nDims; ++dIdx){
      cout << outputLayer[dIdx] << ", ";
   }
   cout << endl;
   return outputLayer;
}

void NRBM::setSettings(NRBMSettings& s){   
   settings = s;
   
   //initialize random generator
   random_device rd;
   mt19937 gen(rd());
   //get a nice normal distribution
   normal_distribution<double> d(0.0002,0.01);
   
   
   //alloc I/O layers
   inputLayer = vector<double>(settings.inputSize, 0);
   outputLayer = vector<double>(settings.outputSize, 0);
   
   //init biases
   biasUp = vector<double>(settings.outputSize, 0);
   biasDown = vector<double>(settings.inputSize, 0);
   
   //init weights using normal distr.
   weights = vector< vector<double> >(settings.inputSize, vector<double>(settings.outputSize));
   for(size_t inIdx = 0; inIdx != settings.inputSize; ++inIdx){
      for(size_t outIdx = 0; outIdx != settings.outputSize; ++outIdx){
         weights[inIdx][outIdx] = d(gen);
      }
   }
   //alloc space to store energies.
   dataEnergy = vector< vector<double> >(settings.inputSize, vector<double>(settings.outputSize, 0));
   modelEnergy = vector< vector<double> >(settings.inputSize, vector<double>(settings.outputSize, 0));
}
