#include <csvm/csvm_mlp.h>

using namespace std;
using namespace csvm;

void MLPerceptron::setSettings(MLPSettings& s){
   this->settings = s;
}

void MLPerceptron::train(vector<Feature>& randomFeatures){
   cout << "train mlp\n";
}

vector<double> MLPerceptron::getActivations(vector<Feature>& imageFeatures){
   cout << "get activation vector from image patches\n";
   return vector<double>();
}