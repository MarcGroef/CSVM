#ifndef PARAM
#define PARAM

using namespace std;

#define NR_FEATURES 6

typedef struct data_struct DATA;
struct data_struct {
int tot_data;
double class1;
double class2;
double input[NR_FEATURES];
};

#endif // PARAM
