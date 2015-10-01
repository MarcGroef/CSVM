
#ifndef CSVM_LBP_DECRIPTOR_H
#define CSVM_LBP_DESCRIPTOR_H

#include <vector>
#include <iostream>
#include <cmath>
#include "csvm_patch.h"
#include "csvm_image.h"
#include "csvm_feature.h"

using namespace std;


namespace csvm {

	class LBPDescriptor {
		

	public:
		LBPDescriptor();
		vector< int > getLBP(Patch patch, int channel);

	};


}


#endif
