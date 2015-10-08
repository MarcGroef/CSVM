# ifndef NUMPER_H
# define NUMPER_H

# include <string>
# include <iostream>
# include <sstream>
# include "param_letters.h"
# include "List.h"
# include "SIMUL.h"

class SVM {
    public:
        SVM( const char * ) ;
        ~SVM() ;
        double run() ;
    
    private:
        double runExperiment() ;
        void alpha_initialize(int, double) ;
        void y_initialize(DATA*) ;
        void compute_kernel_matrix(DATA*) ;
        void compute_test_kernel_matrix(DATA*,DATA*) ;
        void train(int);
        double compute_train_error(DATA*, int, int);
        double compute_test_error(DATA*, int, DATA*);
        double compute_similarity(double*, double*, int);
        void compute_bias(int) ;
        void readParameterFile( ) ;
        double **alpha_coeff;    
        double **kernel_train;
        double **kernel_test;
        double **y;
        double *bias;
        double *output_SVM;
        const char * parameterFile ;
        const char * trainFile ;
        const char * testFile ;    
        std::string parameterPath ;
	double alpha; 
        double beta;
        double COST;
        double D2;
        double SVM_C;
        double EPS;
        double SIGMA;
        double INIT_ALPHA;
	int ALPHA_ITER;
        int NR_REP1;
        int NR_REP2;
        int nHiddenV; 
};

double runSVM(char const *file);
# endif //NUMPER_H
