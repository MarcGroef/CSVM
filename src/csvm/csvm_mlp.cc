#include <csvm/csvm_mlp.h>

//Neuron
//input connections, output connections, threshold, activation function, type
//desired output
//backprogation algorithm
//input vector

using namespace std;
using namespace csvm;
double* inputValue;
double weights[][];

void MLPerceptron::setSettings(MLPSettings& s){
   this->settings = s;
   cout << "settings set\n";
}


void MLPerceptron::train(vector<Feature>& randomFeatures){
   cout << "train mlp\n";
}

vector<double> MLPerceptron::getActivations(vector<Feature>& imageFeatures){
   cout << "get activation vector from image patches\n";
   return vector<double>(10,0);
}

vector<Neuron> createLayer(double* inputWeights, int size){
		vector<Neuron> layer;
		
		p = new (nothrow) int[i];
		if (p == nullptr)
			cout << "Error: memory could not be allocated";
			
		for(int i = 0;i<size;i++){
		Neuron neuron = Neuron(input);
	}
}
//-----start------ Forward propegation
void sumInputUnits(double* inputValue, double* inputWeights, double b){
	double sumInputUnitsTimesWeights = 0;
	b = inputWeights[0];
	for(int i=0;i<inputValue.size();i++)
	{	
    sumInputUnitsTimesWeights+=inputValue[i]*inputWeights[i];
	}
}

void activationFunction(double summedActivation){
	activationValue = 1/(1+exp(-summedActivation));
}
//-----end------ Forward propegation

//-----start------ Backward propegation
double errorFunction(double* desiredOutput, double* actualOutput){
		double error = 0;
		for (int i=0; i< desiredOutput.size();i++){
			error += (desiredOutput[i] - actualOutput[i])*(desiredOutput[i] - actualOutput[i]);
		}
		error *= 0.5;
		return error;
}

//-----end------ Backward propegation
