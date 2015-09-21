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
        SVM( const char *, bool ) ;
        SVM( const char * ) ;
        SVM(){};
        ~SVM() ;
        double run() ;
    
    private:
        double runExperiment() ;
        void alpha_initialize(int, double, bool) ;
        void y_initialize(DATA*) ;
        void y_initialize_regression(DATA*) ;
        void compute_kernel_matrix(DATA*) ;
        void train(int);
        void train_regression(int, int);
        double compute_train_error(DATA*, int, int);
        double compute_train_error_regression(DATA*, int, int);
        double compute_test_error(DATA*, int, DATA*);
        double compute_test_error_regression(double*, double, double**, int);
        double compute_similarity(double*, double*, int, int, double);
        void compute_bias(int) ;
        void compute_bias_regression(int) ;
        void readParameterFile(bool) ;
        double **alpha_coeff;    
        double **alpha_coeff_star;    
        double **kernel_train;
        double **y;
        double *bias;
        double *output_SVM;
        const char * parameterFile ;
        const char * trainFile ;
        const char * testFile ;    
        std::string parameterPath ;

        //Generates a random double between .5 and -.5
        double drand() const;

        //Initialize layer with random values between .5 and -.5
//        void initializeFeatureLayer(double **, int) const;
        void compute_features(double **, int, int);

        //Computer kernel matrix for featureLayer
        void compute_feature_matrix(double**, int numData);
        
        //propagate alpha back to feature layer
        void alpha_back_propagation(double **, int, int);
        
        void update_feature_y(SVM*, double**, int);
        
        void compute_result(double *, DATA*);
        
        void allocate_datamembers(int);

        double alpha; 
        double Falpha;
        double beta;
        double COST;
        double FCOST;
        double D2;
        double FD2;
        double SVM_C;
        double FSVM_C;
        double Y_FRACT;
        double Y_INIT;
        double EPS;
        double FEPS;
        double SIGMA;
        double FSIGMA;
        double INIT_ALPHA;
        double INIT_ALPHA_1;
        double INIT_ALPHA_2;
        double FINIT_ALPHA;
        double FALPHA_ADDSTART;
        int EPOCHS;
        int FLAYER_SIZE;
	
        int ALPHA_ITER;
        int ALPHA_ITER_INIT;
        int FALPHA_ITER;
        int NR_REP1;
        int NR_REP2;
        int nHiddenV; 
};

double runSVM(char const *file);
# endif //NUMPER_H
