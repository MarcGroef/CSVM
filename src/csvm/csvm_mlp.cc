#include <csvm/csvm_mlp.h>

//Neuron
//input connections, output connections, threshold, activation function, type
//desired output
//backprogation algorithm
//input vector

using namespace std;
using namespace csvm;

void MLPerceptron::setSettings(MLPSettings& s){
   this->settings = s;
   cout << "settings set\n";
}


void MLPerceptron::train(vector<Feature>& randomFeatures){
	int nInputNodes = randomFeatures.at(0).size;
	int nHiddenNodes = (inputNodes + outputNodes)/2;
}

vector<double> MLPerceptron::getActivations(vector<Feature>& imageFeatures){
   //cout << "get activation vector from image patches\n";
   //cout << imageFeatures.size;
   return vector<double>(10,0);
}
