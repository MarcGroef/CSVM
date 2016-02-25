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
}
