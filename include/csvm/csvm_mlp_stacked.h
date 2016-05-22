#ifndef CSVM_MLP_STACKED_H
#define CSVM_MLP_STACKED_H

#include <iostream>
#include <vector>
#include "csvm_mlp.h"
#include "csvm_settings.h"

using namespace std;

namespace csvm{
	
	class MLPStacked{
	
	private:
	  
	  MLPSettings settings;
	  
	  MLPerceptron mlp;
	  
	  vector<double> highestHiddenActivationImage(vector<Feature> imageFeatures);
		
	public:

	
	void setSettings(MLPSettings s);
	
	
	};
}
#endif
