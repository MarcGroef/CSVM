
#ifndef CSVM_LBP_DESCRIPTOR_H
#define CSVM_LBP_DESCRIPTOR_H

//DEPRECATED

#include <vector>
#include <iostream>
#include <cmath>
#include "csvm_patch.h"
#include "csvm_image.h"
#include "csvm_feature.h"

#include <bitset>


using namespace std;


namespace csvm {

	class LBPDescriptor {
		vector<int> featureIndex;
	public:
		LBPDescriptor();
		Feature getLBP(Patch patch, int channel);
	private:
		bool isUniform(int lbp);
		int uniformValue(int lbp);
	};

	


}


#endif
