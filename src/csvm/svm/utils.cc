# include "utils.h"


using namespace std;

string prevFnName = "";

// Standard output/logger wrapper
void report(int cLev, string functionName, string message){
	if (functionName != prevFnName) {
		cout << endl;
		prevFnName = functionName;
	}
	for (int i=0; i<cLev; i++)		cout << "\t";
	cout << functionName << ": " << message << endl;
}


