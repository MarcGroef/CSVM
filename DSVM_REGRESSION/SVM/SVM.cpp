# ifndef NUMPER
# define NUMPER

# include <math.h>
# include <stdio.h>
# include <string.h>
# include <stdlib.h>
# include <fstream>
# include <vector>
# include <clocale>
# include <float.h>
# include <iostream>
# include <sys/time.h>

# include "SIMUL.h"							// includes structs for SIMULATIONS en RESULTS
# include "SVM.h"							// overview of SVM class
# include "param_letters.h"					// DATA struct with hardcoded nr. of features

#define CLASSES 1
#define BETA_ITER 1
#define AVERAGE_OUT 22.5

//#define KERNEL 3

#define LINEAR 1							// for Kernel selection
#define TANH 2
#define GAUSS 3

#define NR_DISTANCES 1

extern DATA *read_data(char *);				// defined in param_letters.c
extern double fabs(double);					// downgrading float Math::fabs()
extern SIMULATIONS *insert_simulation(SIMULATIONS*, long int, double);
extern void show_simulations(SIMULATIONS *);
extern void write_to_file_all(char*,SIMULATIONS*);

using namespace std;

bool SVM_verbose = false;




double sqr(double x)
{
    return x * x;
}





//added KERNEL argument instead of using a defined kernel so we can destinquish between the feature and main kernel
double SVM::compute_similarity(double *x, double *y, int actionDimension, int KERNEL, double SIG)
{
    double input = 0.0;
   // GPU mogelijkheid
    for (int ad = 0; ad < actionDimension; ++ad)
        input += x[ad] * y[ad];
    if (KERNEL == LINEAR)
        return input;
    else if (KERNEL == TANH)
    {
        double output = -1;
        if (input > 1.92032) 
	        output = 0.96016;
        else if (input > 0.0)
            output = -0.260373271 * input * input + input;
        else if (input > -1.92032)
            output = 0.260373271 * input * input + input;
        else
	        output = -0.96016;
        return output;
    }
    else if (KERNEL == GAUSS)
    {
        double diff = 0.0;
        for (int ad = 0; ad < actionDimension; ++ad)
            diff += (x[ad] - y[ad]) * (x[ad] - y[ad]);
        double output = exp(-diff / SIG);
        return output;
    }

    cout << "*** NO KERNEL SELECTED " << endl;
}



double exact_Tan(double tot_F)
{
    return  2.0 / (1.0 + exp(-tot_F)) - 1.0;
}





SVM::SVM(const char * pFile, bool featureSVM)
{
    parameterFile = pFile;
    parameterPath.assign(parameterFile);
    readParameterFile(featureSVM);
}




SVM::SVM(const char * pFile)
{
    parameterFile = pFile;
    parameterPath.assign(parameterFile);
    readParameterFile(false);
}



SVM::~SVM()
{
}





void SVM::y_initialize_regression(DATA *leerdata)
{
    for (int episode = 0; episode < leerdata[0].tot_data; ++episode) 
    {
        double label =  leerdata[episode].class1;
        y[0][episode] = label;
    }
}   
        




void SVM::y_initialize(DATA *leerdata)
{
    for (int episode = 0; episode < leerdata[0].tot_data; ++episode) 
    {
        int label = (int) leerdata[episode].class1;
        for(int C = 0 ; C < CLASSES ; C++)
            if (C == label)
                y[C][episode] = 1.0;
            else
                y[C][episode] = -1.0;
    }
}   





void SVM::alpha_initialize(int tot_data, double SVM_C, bool feature = false)
{
    // cout << "init random = " << feature << endl;

    for (int C = 0; C < CLASSES; ++C)
        for (int episode = 0; episode < tot_data; ++episode)
        {
            if (feature)
            {
                alpha_coeff[C][episode] =  (INIT_ALPHA_1 + INIT_ALPHA_2 * drand48()) * SVM_C * INIT_ALPHA; // (INIT_ALPHA is between 0 and 1)
                alpha_coeff_star[C][episode] =  (INIT_ALPHA_1 + INIT_ALPHA_2 * drand48()) * SVM_C * INIT_ALPHA; // (INIT_ALPHA is between 0 and 1)
                //alpha_coeff[C][episode] =  (drand48()) * SVM_C * INIT_ALPHA; // (INIT_ALPHA is between 0 and 1)
                //alpha_coeff_star[C][episode] =  (drand48()) * SVM_C * INIT_ALPHA; // (INIT_ALPHA is between 0 and 1)
            } else
            {
                alpha_coeff[C][episode] =  SVM_C * INIT_ALPHA; // (INIT_ALPHA is between 0 and 1)
                alpha_coeff_star[C][episode] =  SVM_C * INIT_ALPHA; // (INIT_ALPHA is between 0 and 1)
            }
        }
}
            




void SVM::compute_kernel_matrix(DATA *leerdata)
{
    for (int episode = 0; episode < leerdata[0].tot_data; ++episode)
        for (int ad = 0; ad < leerdata[0].tot_data; ++ad)
            kernel_train[episode][ad] = compute_similarity(leerdata[episode].input,leerdata[ad].input,NR_FEATURES, GAUSS, SIGMA);
}





void SVM::compute_feature_matrix(double **featureLayer, int numData)
{
    for (int episode = 0; episode < numData; ++episode)
        for (int ad = 0; ad < numData; ++ad)
            kernel_train[episode][ad] = compute_similarity(featureLayer[episode],featureLayer[ad],FLAYER_SIZE, GAUSS, SIGMA);
}






void SVM::alpha_back_propagation(double **featureLayer, int tot_data, int init)
{
    int kernel = GAUSS;
  if (init == 1)
    {
     for (int feature = 0; feature != FLAYER_SIZE; ++feature)
         for (int episode = 0; episode != tot_data; ++episode)
               featureLayer[episode][feature] = y[0][episode] * Y_FRACT * (Y_INIT * drand48() + (1.0 - Y_INIT));
     }
  else
   {
    if (kernel == LINEAR)
    {
        for (int C = 0; C != CLASSES; ++C)
        {
            for (int feature = 0; feature != FLAYER_SIZE; ++feature)
            {
                double sum = 0.0;
                for (int episode = 0; episode != tot_data; ++episode)
                    sum += (alpha_coeff_star[0][episode] - alpha_coeff[0][episode]) * featureLayer[episode][feature];
                    
                for (int episode = 0; episode != tot_data; ++episode)
                {
                    featureLayer[episode][feature] += beta * sum * (alpha_coeff_star[0][episode] - alpha_coeff[0][episode]);
                    
                    if (featureLayer[episode][feature] > 1)
                        featureLayer[episode][feature] = 1;
                    else if (featureLayer[episode][feature] < -1)
                        featureLayer[episode][feature] = -1;
                }
            }
        }
    }

    if (kernel == GAUSS)
    {
        for (int C = 0; C != CLASSES; ++C)
        {
            double **bufferLayer = new double*[tot_data];
            for (int episode = 0; episode != tot_data; ++episode)
                bufferLayer[episode] = new double[FLAYER_SIZE];
                
            for (int feature = 0; feature != FLAYER_SIZE; ++feature)
            {
                for (int episode = 0; episode != tot_data; ++episode)
                {
                    double sum = 0.0;
                    for (int ad = 0 ; ad != tot_data; ++ad)
                    {
                        double epower = kernel_train[episode][ad];
                            
                        sum += (alpha_coeff_star[0][episode] - alpha_coeff[0][episode]) * 
                            (alpha_coeff_star[0][ad] - alpha_coeff[0][ad]) * 
                            (featureLayer[episode][feature] - featureLayer[ad][feature]) / SIGMA * 
                            epower;
                    }
                    bufferLayer[episode][feature] = -beta * sum;
                }
            }
            for (int feature = 0; feature != FLAYER_SIZE; ++feature)
            {
                for (int episode = 0; episode != tot_data; ++episode)
                {
                    if (featureLayer[episode][feature] + bufferLayer[episode][feature] >= 1)
                        featureLayer[episode][feature] = 1;
                    else if (featureLayer[episode][feature] + bufferLayer[episode][feature] <= -1)
                        featureLayer[episode][feature] = -1;
                    else
                        featureLayer[episode][feature] += bufferLayer[episode][feature];
                }
            }
            for (int episode = 0; episode != tot_data; ++episode)
                delete[] bufferLayer[episode];
            delete[] bufferLayer;
        }
    }
  }
}






void SVM::update_feature_y(SVM *featureSVM, double **featureLayer, int tot_data)
{
    for (int flayer = 0; flayer != FLAYER_SIZE; ++flayer)
        for (int C = 0; C != CLASSES; ++C)
            for (int episode = 0; episode != tot_data; ++episode)
                featureSVM[flayer].y[C][episode] = featureLayer[episode][flayer];
}






void SVM::train_regression(int tot_data, int init)
{
    double QTarget1, delta_alpha, delta_alpha_star;
    int nr_iter;

    if (init == 1)
       nr_iter = ALPHA_ITER_INIT;
    else
       nr_iter = ALPHA_ITER;
    for (int C = 0; C < CLASSES; ++C)
    {
        for (int alpha_iter = 0; alpha_iter < nr_iter; ++alpha_iter)
        {
            for (int episode = 0; episode < tot_data; ++episode)
            {
                double sum_1 = 0.0;
                double sum_2 = 0.0;
                double sum_3 = 0.0;
                for (int i = 0; i < tot_data; ++i)
                    sum_1 += (alpha_coeff_star[C][i] - alpha_coeff[C][i]) * kernel_train[episode][i];

                delta_alpha = (-1.0 * EPS - y[C][episode] + sum_1); // * 10.0 / y[C][episode];
                delta_alpha_star = (-1.0 * EPS + y[C][episode] - sum_1); //

                QTarget1 = alpha_coeff[C][episode] + delta_alpha * alpha; // * 20.0 / y[C][episode]; // - 2.0 * COST * sum_2;
                if (QTarget1 > SVM_C)
                    QTarget1 = SVM_C;
                if (QTarget1 < 0)
                    QTarget1 = 0;
                alpha_coeff[C][episode] = QTarget1;

                QTarget1 = alpha_coeff_star[C][episode] + delta_alpha_star * alpha; // * 20.0 / y[C][episode]; // - 2.0 * COST * sum_2;
                if (QTarget1 > SVM_C)
                    QTarget1 = SVM_C;
                if (QTarget1 < 0)
                    QTarget1 = 0;
                alpha_coeff_star[C][episode] = QTarget1;

                delta_alpha = 0.0;
                delta_alpha_star = 0.0;
                //for (int rep_corr = 0; rep_corr < 5; ++rep_corr)
                for (int rep_corr = 0; rep_corr < 1; ++rep_corr)
                {
                    sum_2 = 0.0;
                    for (int i = 0; i < tot_data; ++i)
                        sum_2 += alpha_coeff_star[C][i] - alpha_coeff[C][i];
                    delta_alpha +=  0.1 * (+2.0 * COST * sum_2);
                    delta_alpha_star += 0.1 * (-2.0 * COST * sum_2);
                    QTarget1 = alpha_coeff[C][episode] + delta_alpha * alpha;
                    if (QTarget1 > SVM_C)
                        QTarget1 = SVM_C;
                    if (QTarget1 < 0)
                        QTarget1 = 0;
                    alpha_coeff[C][episode] = QTarget1;

                    QTarget1 = alpha_coeff_star[C][episode] + delta_alpha_star * alpha ; // * 20.0 / y[C][episode]; // - 2.0 * COST * sum_2;
                    if (QTarget1 > SVM_C)
                        QTarget1 = SVM_C;
                    if (QTarget1 < 0)
                        QTarget1 = 0;
                    alpha_coeff_star[C][episode] = QTarget1;
                }
            }
            for (int rep_corr_mult = 0; rep_corr_mult < 10; ++rep_corr_mult)
            {
                for (int i = 0; i < tot_data; ++i)
                {
                    delta_alpha = 0.2 * (- 2.0 * D2 * alpha_coeff_star[C][i]); // * alpha_coeff[C][episode]);
                    delta_alpha_star = 0.2 * (- 2.0 * D2 * alpha_coeff[C][i]); // * alpha_coeff_star[C][episode]);
                    QTarget1 = alpha_coeff[C][i] + delta_alpha * alpha; // * 20.0 / y[C][episode]; // - 2.0 * COST * sum_2;
                    if (QTarget1 > SVM_C)
                        QTarget1 = SVM_C;
                    if (QTarget1 < 0)
                         QTarget1 = 0;
                    alpha_coeff[C][i] = QTarget1;
                    QTarget1 = alpha_coeff_star[C][i] + delta_alpha_star * alpha; // * 20.0 / y[C][episode]; // - 2.0 * COST * sum_2;
                    if (QTarget1 > SVM_C)
                        QTarget1 = SVM_C;
                    if (QTarget1 < 0)
                        QTarget1 = 0;
                    alpha_coeff_star[C][i] = QTarget1;
                }
            }
        }
    }
}
 






void SVM::train(int tot_data)
{
    double delta_alpha;
    double target;

     for (int alpha_iter = 0; alpha_iter < ALPHA_ITER; ++alpha_iter)
       for (int episode = 0; episode < tot_data; ++episode)
           for (int C = 0; C < CLASSES; ++C)
                 { 
                   double sum_1 = 0.0;
                   for (int i = 0; i < tot_data; ++i)
                       {
                                    sum_1 += alpha_coeff[C][i] * y[C][episode] * y[C][i] * kernel_train[episode][i];
                       }
	
				
                   delta_alpha = 1.0 -  sum_1; // - 2 * COST * sum_2 * y[C][episode];
                   target = alpha_coeff[C][episode] + delta_alpha * alpha; //  * TrainResult2[episode];

                        if (target > SVM_C)
                            target = SVM_C;
                        if (target < 0)
                            target = 0;
                        alpha_coeff[C][episode] = target;
                  } 
            for(int rep_corr1 = 0 ; rep_corr1 < 5 ; rep_corr1++)            
	     for (int episode = 0; episode < tot_data; ++episode)
              {
                    for (int C = 0; C < CLASSES; ++C)
                     { 
                        double sum_2 = 0.0;
                       /* GPU mogelijkheid */	
                       for(int i = 0 ; i < tot_data ; i++)
                          sum_2 += alpha_coeff[C][i] * y[C][i];
                       delta_alpha = - 2 * COST * sum_2 * y[C][episode];
                       target = alpha_coeff[C][episode] + delta_alpha * alpha; //  * TrainResult2[episode];
                       if (target > SVM_C)
                           target = SVM_C;
                       if (target < 0)
                           target = 0;
                       alpha_coeff[C][episode] = target;
                     }
                }
}
           




void SVM::compute_bias_regression(int tot_data)
{
    for (int C = 0; C < CLASSES; ++C)
    {
        bias[C] = 0;
        int tot = 0;
        for (int i = 0; i < tot_data; ++i)
        {
            if ((alpha_coeff[C][i] > -0.0001) && ((alpha_coeff[C][i]) < (SVM_C + 0.00001)))
            {
                output_SVM[C] = 0.0;
                for (int j = 0; j < tot_data; ++j)
                {
                    double tot_F =  kernel_train[j][i]; 
                    output_SVM[C] += (alpha_coeff_star[C][j] - alpha_coeff[C][j]) *  tot_F;
                }
                bias[C] += y[C][i] - output_SVM[C];
                ++tot;
            }
        }
        if (tot != 0)
            bias[C] /= tot;
        else
            bias[C] = 0;
    }
}







void SVM::compute_bias(int tot_data)
{
        /* Bereken Bias van NSVM */
               for (int C = 0; C < CLASSES; ++C)
                {
                    bias[C] = 0;
                    int tot = 0;
                    for (int i = 0; i < tot_data; ++i)
                    {
                        if ((alpha_coeff[C][i] > 0.0) && ((alpha_coeff[C][i]) < (SVM_C - 0.000001)))
                        {
	                        output_SVM[C] = 0.0;
	                        for (int j = 0; j < tot_data; ++j)
	                        {
	                            double tot_F =  kernel_train[j][i]; 
                                    output_SVM[C] += alpha_coeff[C][j] * y[C][j] * tot_F;
	                        }
                                     //bias[C] += y[C][i] - output_SVM[C]; 
	                             //tot++;
                                     if ((y[C][i] * output_SVM[C]) < 0.1)
                                      {
                                         bias[C] += y[C][i] - output_SVM[C];
	                               tot++;
                                     }
	                    }
                    }
                    if (tot == 0)
                     bias[C] = 0.0;
                   else
                      bias[C] /= tot;
                } 
}

                






void SVM::compute_features(double **featureLayer, int feature, int tot_data)
{
    for (int C = 0; C != CLASSES; ++C)
    {
        for (int i = 0; i != tot_data; ++i)
        {
            output_SVM[C] = 0.0;
            for (int j = 0; j < tot_data; ++j)
            {
                double tot_F =  kernel_train[i][j];
                output_SVM[C] += (alpha_coeff_star[C][j] - alpha_coeff[C][j]) *  tot_F;
            }
            featureLayer[i][feature] = output_SVM[C] + bias[C];
        }
    }
}






void SVM::compute_result(double *input, DATA *leerdata)
{
    for (int C = 0; C < CLASSES; ++C)
    {
        output_SVM[C] = 0.0;
        for (int j = 0; j < leerdata[0].tot_data; ++j)
        {
            /*
            double dist = 0.0;
            for(int i = 0 ; i < NR_FEATURES ; i++)
                dist += fabs(pow(input[i] - leerdata[j].input[i], 2)); 
            
            double tot_F = exp(-dist / SIGMA);
            */
            double tot_F = compute_similarity(leerdata[j].input, input, NR_FEATURES, GAUSS, SIGMA);
            output_SVM[C] += (alpha_coeff_star[C][j] - alpha_coeff[C][j]) *  tot_F;
        }
        output_SVM[C] += bias[C];
    } 
}






double SVM::compute_train_error_regression(DATA *train_example, int first_id, int tot_data)
{
    double label;

    label = train_example[0].class1;
    for (int C = 0; C < CLASSES; ++C)
    {
        output_SVM[C] = 0.0;
        for (int j = 0; j < tot_data; ++j)
        {
            double tot_F =  kernel_train[first_id][j];
            output_SVM[C] += (alpha_coeff_star[C][j] - alpha_coeff[C][j]) *  tot_F;
         }
         output_SVM[C] += bias[C];
    }

    double error = 0.0;
    for (int C = 0; C < CLASSES; ++C)
        error += sqr(output_SVM[C] - label);

    return error;
}






//double SVM::compute_test_error_regression(double *input, double label, DATA *leerdata)
double SVM::compute_test_error_regression(double *input, double label, double **leerdata, int tot_data)
{
    int KERNEL = GAUSS;
    double tot_F;
    for (int C = 0; C < CLASSES; ++C)
    {
        output_SVM[C] = 0.0;
        for (int j = 0; j < tot_data; ++j)
        {
            double dist = 0.0;
            if (KERNEL == LINEAR)
	    {
   	    for(int i = 0 ; i < FLAYER_SIZE ; i++)
                dist += input[i] * leerdata[j][i]; 
            tot_F = dist;
	    }
	    else if (KERNEL == GAUSS)
	    {
	     for(int i = 0 ; i < FLAYER_SIZE ; i++)
                dist += fabs(pow(input[i] - leerdata[j][i], 2)); 
            tot_F = exp(-dist / SIGMA);
	    }
            
            output_SVM[C] += (alpha_coeff_star[C][j] - alpha_coeff[C][j]) *  tot_F;
        }
        output_SVM[C] += bias[C];
    }
//    cout << "outputSVM: " << output_SVM[0] << '\n';
    
    double error = 0.0;
    for (int C = 0; C < CLASSES; ++C)
        error += sqr(output_SVM[C] - label);

//    cout << "error: " << error << '\n';
     return error;
}











/*
double SVM::compute_test_error_regression(DATA *testdata, int episode2, DATA *leerdata)
{
    double label;

    label = testdata[episode2].class1;
    

    for (int C = 0; C < CLASSES; ++C)
         {
          output_SVM[C] = 0.0;
          for (int j = 0; j < leerdata[0].tot_data; ++j)
             {
              double dist = 0.0;
              for(int i = 0 ; i < NR_FEATURES ; i++)
                   dist += fabs(pow(testdata[episode2].input[i] - leerdata[j].input[i], 2)); 
              double tot_F = exp(-dist / SIGMA);
              output_SVM[C] += (alpha_coeff_star[C][j] - alpha_coeff[C][j]) *  tot_F;
             }
            output_SVM[C] += bias[C];
          } 

      double error = 0.0;
      for (int C = 0; C < CLASSES; ++C)
        error += sqr(output_SVM[C] - label);

     return error;
}
*/














/* TEST op trainset */ 
double SVM::compute_train_error(DATA *train_example, int first_id, int tot_data)
{
    int label = (int) train_example[0].class1;
    double TrainResult;
    double *rewardsPerClass    = new double[CLASSES];

    for (int C = 0; C < CLASSES; ++C)
    {
        output_SVM[C] = 0.0;
        for (int j = 0; j < tot_data; ++j)
        {
            double tot_F =  kernel_train[first_id][j];
            output_SVM[C] += alpha_coeff[C][j] * y[C][j] * tot_F;
        }
        output_SVM[C] += bias[C];

    } 
    for (int C = 0; C < CLASSES; ++C)
    rewardsPerClass[C] = output_SVM[C];
    
    double maxRewardPerClass = rewardsPerClass[0] ;
    int nMax = 1 ;

    for ( int C = 1 ; C < CLASSES ; C++ ) 
    {
        if ( rewardsPerClass[C] > maxRewardPerClass ) 
        {
            maxRewardPerClass = rewardsPerClass[C] ;
            nMax = 1 ;
        } else if ( rewardsPerClass[C] == maxRewardPerClass ) {
            nMax++ ;
        }
    }
    if ( rewardsPerClass[ label ] == maxRewardPerClass )
        TrainResult = 1.0 - 1.0/nMax ;
    else
        TrainResult= 1.0 ;

    if ( rewardsPerClass[ label ] == maxRewardPerClass )
        TrainResult= 1.0 - 1.0/nMax ;
    else
        TrainResult= 1.0 ;

    return TrainResult;
}





	            
double SVM::compute_test_error(DATA *testdata, int episode2, DATA *leerdata)
{
    double F_episode;
    double TrainResult;
    double *rewardsPerClass    = new double[CLASSES];

    int label = (int) testdata[episode2].class1;
    for (int C = 0; C < CLASSES; ++C)
    {
        output_SVM[C] = 0.0;
        for (int j = 0; j < leerdata[0].tot_data; ++j)
        {
            double dist = 0.0;
            for(int i = 0 ; i < NR_FEATURES ; i++)
                dist += fabs(pow(testdata[episode2].input[i] - leerdata[j].input[i], 2)); 
            F_episode = exp(-dist / SIGMA);
            output_SVM[C] += alpha_coeff[C][j] * y[C][j] * F_episode;
        }
        output_SVM[C] += bias[C];
    } 
    for (int C = 0; C < CLASSES; ++C)
        rewardsPerClass[C] = output_SVM[C];
            
    double maxRewardPerClass = rewardsPerClass[0] ;
    int nMax = 1 ;

    for ( int C = 1 ; C < CLASSES ; C++ ) 
    {
        if ( rewardsPerClass[C] > maxRewardPerClass ) 
        {
            maxRewardPerClass = rewardsPerClass[C] ;
            nMax = 1 ;
        } else if ( rewardsPerClass[C] == maxRewardPerClass ) 
            nMax++ ;

        if ( rewardsPerClass[ label ] == maxRewardPerClass )
            TrainResult = 1.0 - 1.0/nMax ;
        else
            TrainResult= 1.0 ;
    }
    if ( rewardsPerClass[ label ] == maxRewardPerClass )
        TrainResult= 1.0 - 1.0/nMax ;
    else
        TrainResult= 1.0 ;
    return TrainResult;
}

double SVM::drand() const
{
    return static_cast<double>(rand()) / RAND_MAX - .5;
}

double SVM::run()
{
    double AverageresultPerEpisode = runExperiment();
	
    return AverageresultPerEpisode;
}





void SVM::allocate_datamembers(int num_data)
{
    bias = new double[CLASSES];
    output_SVM = new double[CLASSES];

    y = new double*[CLASSES];
    alpha_coeff = new double*[CLASSES];
    alpha_coeff_star = new double*[CLASSES];

    kernel_train = (double **) calloc(num_data,sizeof(double*));
    for(int i = 0 ; i < num_data ; i++)
        kernel_train[i] = new double[num_data];
    
    for (int C = 0; C < CLASSES; ++C)
    {
        y[C] = new double[num_data];
        alpha_coeff[C] = new double[num_data];
        alpha_coeff_star[C] = new double[num_data];
    }
}




double SVM::runExperiment()
{       
    DATA *data;
    SIMULATIONS *SIMULTRAIN;
    SIMULATIONS *SIMULTEST;
    SIMULATIONS *SIMULBEST;
    char added[150];
    char fileout[250];
    
    SIMULTRAIN = NULL;
    SIMULTEST = NULL;
    SIMULBEST = NULL;
    
    int i;
    int j;
            
    double *rewardsPerClass    = new double[CLASSES];
    double *totrewardsPerClass = new double[CLASSES];

    data = NULL;
    double best_train;
    double best_test;
    int best_found;

    data = read_data("28.csv");
    int num_data = data[0].tot_data + 1;
    DATA *leerdata = new DATA[num_data];
    DATA *testdata = new DATA[num_data];

    allocate_datamembers(num_data);

    SVM *featureSVM = new SVM[FLAYER_SIZE];
    for (int fnode = 0; fnode != FLAYER_SIZE; ++fnode)
    {
        featureSVM[fnode] = SVM(parameterFile, true);
        featureSVM[fnode].allocate_datamembers(num_data);
    }
    
    for (int rep = 0; rep < NR_REP1; ++rep)
    {
        for (int C = 0; C < CLASSES; ++C)
            bias[C] = 0;

        if (data != NULL)
            free(data);
    data = read_data("28.csv");

        best_train = DBL_MAX;
        leerdata[0].tot_data = 0;
        testdata[0].tot_data = 0;
        for (i = 0; i < data[0].tot_data; ++i)
        {
            if (leerdata[0].tot_data >= (data[0].tot_data * 9 / 10))
            {
                testdata[testdata[0].tot_data].class1 = data[i].class1;
                for (j = 0; j < NR_FEATURES; ++j)
                    testdata[testdata[0].tot_data].input[j] = data[i].input[j];
                ++testdata[0].tot_data;
            }
            else if (testdata[0].tot_data >= (data[0].tot_data * 1 / 10))
            {
                leerdata[leerdata[0].tot_data].class1 = data[i].class1;
                for (j = 0; j < NR_FEATURES; ++j)
                    leerdata[leerdata[0].tot_data].input[j] = data[i].input[j];
                ++leerdata[0].tot_data;
            }
            else if (drand48() < (1.0 / 10))
            {
                testdata[testdata[0].tot_data].class1 = data[i].class1;
                for (j = 0; j < NR_FEATURES; ++j)
                    testdata[testdata[0].tot_data].input[j] = data[i].input[j];
                ++testdata[0].tot_data;
            }
            else
            {
                leerdata[leerdata[0].tot_data].class1 = data[i].class1;
                for (j = 0; j < NR_FEATURES; ++j)
                    leerdata[leerdata[0].tot_data].input[j] = data[i].input[j];
                ++leerdata[0].tot_data;
            }
        }
        
        // INITIALIZE

        double **featureLayer = new double*[leerdata[0].tot_data];
        for (int feature = 0; feature != leerdata[0].tot_data; ++feature)
            featureLayer[feature] = new double[FLAYER_SIZE];
        
        y_initialize_regression(leerdata);
        
        alpha_initialize(leerdata[0].tot_data,SVM_C);
        for (int fnode = 0; fnode != FLAYER_SIZE; ++fnode)
        {
            featureSVM[fnode].alpha_initialize(leerdata[0].tot_data,FSVM_C, true);
            featureSVM[fnode].compute_kernel_matrix(leerdata);
            featureSVM[fnode].compute_features(featureLayer, fnode, leerdata[0].tot_data);
        }

        compute_feature_matrix(featureLayer, leerdata[0].tot_data);
        alpha_back_propagation(featureLayer, leerdata[0].tot_data, 1);
        update_feature_y(featureSVM, featureLayer, leerdata[0].tot_data);
        for(int flayer = 0; flayer != FLAYER_SIZE; ++flayer)
              {
                  featureSVM[flayer].train_regression(leerdata[0].tot_data, 1);
                  featureSVM[flayer].compute_bias_regression(leerdata[0].tot_data);
                  featureSVM[flayer].compute_features(featureLayer, flayer, leerdata[0].tot_data);
              }
        
        for (int rep2 = 0; rep2 <= NR_REP2; ++rep2)
        {
            /*
            train(leerdata[0].tot_data);
            compute_bias(leerdata[0].tot_data); 
            */
            
            //FIRST MAIN SVM RUN    
            compute_feature_matrix(featureLayer, leerdata[0].tot_data);
            train_regression(leerdata[0].tot_data,0);
            compute_bias_regression(leerdata[0].tot_data);
            
            //TRAINING
            for (int epoch = 0; epoch != EPOCHS; ++epoch)
            {
                alpha_back_propagation(featureLayer, leerdata[0].tot_data, 0);
                update_feature_y(featureSVM, featureLayer, leerdata[0].tot_data);
                compute_feature_matrix(featureLayer, leerdata[0].tot_data);

                for(int flayer = 0; flayer != FLAYER_SIZE; ++flayer)
                {
                    featureSVM[flayer].train_regression(leerdata[0].tot_data,0);
                    featureSVM[flayer].compute_bias_regression(leerdata[0].tot_data);
                    featureSVM[flayer].compute_features(featureLayer, flayer, leerdata[0].tot_data);
                }
                
                compute_feature_matrix(featureLayer, leerdata[0].tot_data);
                
                train_regression(leerdata[0].tot_data,0);

                compute_bias_regression(leerdata[0].tot_data);
            }
            
            double averageRewards = 0;
            for (int e = 0; e < leerdata[0].tot_data; ++e)
                averageRewards += compute_train_error_regression(&leerdata[e], e, leerdata[0].tot_data);
	
            averageRewards /= leerdata[0].tot_data;
 
            SIMULTRAIN = insert_simulation(SIMULTRAIN, rep2, averageRewards);
            if (averageRewards <= best_train) 
            { 
                best_train = averageRewards;
                best_found = 1;
            }
            else 
                best_found = 0;
            // best_found = 1; // BEST AT END 

            if (averageRewards <= best_train)
                best_train = averageRewards;

            //TESTING
            averageRewards = 0;
            for (int e = 0; e < testdata[0].tot_data; ++e)
            {
                double *input = new double[FLAYER_SIZE];
                for (int flayer = 0; flayer != FLAYER_SIZE; ++flayer)
                {
                        featureSVM[flayer].compute_result(testdata[e].input, leerdata);
                        input[flayer] = featureSVM[flayer].output_SVM[0];
      //                  cout << "input " << flayer << ": " << input[flayer] << '\n';
                }
                averageRewards += compute_test_error_regression(input, testdata[e].class1, featureLayer, leerdata[0].tot_data);
//                cout << "averageRewards after " << e << " instances: " << averageRewards << '\n';
            }
            averageRewards /= testdata[0].tot_data;
            
  //          cout << "final average rewards: " << averageRewards << '\n';
            
            SIMULTEST = insert_simulation(SIMULTEST, rep2, averageRewards);

            if (best_found) 
                best_test = averageRewards;
            
            show_simulations(SIMULTRAIN);
            show_simulations(SIMULTEST);
        }

        SIMULBEST = insert_simulation(SIMULBEST, 0, best_test);
        show_simulations(SIMULTRAIN);
        show_simulations(SIMULTEST);
        show_simulations(SIMULBEST);
       /* sprintf(fileout,"Results/");
        strcat(fileout,"DATASET_HOUSING_INITY_GAUSS");
        sprintf(added,"_alpha:%1.11f_beta:%1.7f_COST:%1.3f_D2:%1.1f_SVMC:%1.1f_ALPHAITER:%d_REP1:%d_REP2:%d_EPS:%1.2f",
				alpha,beta,COST,D2,SVM_C,ALPHA_ITER,NR_REP1,NR_REP2,EPS);
        strcat(fileout,added);
          write_to_file_all(fileout,SIMULTEST);
        strcat(fileout,"_BEST");
          write_to_file_all(fileout,SIMULBEST);
         */
       
    }

    delete [] leerdata;
    delete [] testdata;
    delete [] data;

    for (int C = 0; C < CLASSES; ++C)
    {
        delete [] y[C];
        delete [] alpha_coeff[C];
    }

    delete [] rewardsPerClass;
    delete [] totrewardsPerClass ;
    
    return SIMULBEST->average;
}







void SVM::readParameterFile(bool featureSVM)
{
    ifstream ifile;
    
    ifile.open(parameterFile, ifstream::in);
    
    string temp;
    
    while (temp.compare( "algorithm" ) != 0)
	    ifile >> temp;
	 
	ifile >> temp >> alpha;
	ifile >> temp >> Falpha;
	ifile >> temp >> beta;
	ifile >> temp >> COST;
	ifile >> temp >> FCOST;
	ifile >> temp >> D2;
    ifile >> temp >> FD2;
	ifile >> temp >> SVM_C;
	ifile >> temp >> FSVM_C;
	ifile >> temp >> Y_FRACT;
	ifile >> temp >> Y_INIT;
	ifile >> temp >> ALPHA_ITER;
	ifile >> temp >> ALPHA_ITER_INIT;
    ifile >> temp >> FALPHA_ITER;
	ifile >> temp >> NR_REP1;
	ifile >> temp >> NR_REP2;
	ifile >> temp >> EPS;
	ifile >> temp >> FEPS;
	ifile >> temp >> SIGMA;
	ifile >> temp >> FSIGMA;
	ifile >> temp >> INIT_ALPHA;
	ifile >> temp >> INIT_ALPHA_1;
	ifile >> temp >> INIT_ALPHA_2;
    ifile >> temp >> FINIT_ALPHA;
    ifile >> temp >> FALPHA_ADDSTART;
    ifile >> temp >> EPOCHS;
	ifile >> temp >> FLAYER_SIZE;
    
    if (featureSVM)
    {
        alpha = Falpha;
        COST = FCOST;
        D2 = FD2;
        SVM_C = FSVM_C;
        ALPHA_ITER = FALPHA_ITER;
        EPS = FEPS;
        SIGMA = FSIGMA;
        INIT_ALPHA = FINIT_ALPHA;
    }   
}







void seed_random()
{
    struct timeval tv;
    int result = gettimeofday(&tv, NULL);
    int val = tv.tv_sec + tv.tv_usec;
    srand48(val);
}










int main(int argn, char *argv[])
{
    //SVM_verbose = true;
    seed_random();
    SVM * i = new SVM(argv[1]);
    double result = i->run();
    if (result < 0.000000000001)
       result = 100000000000.0; // cout << "WARNING: RESULT IS NAN\n";
    cout << result << endl; 
    delete i;
}








double runSVM(char const *file)
{
    char *oldLocale = setlocale(LC_ALL, NULL);
    SVM_verbose = false;
    setlocale(LC_ALL, "C");
    seed_random();
    SVM * i = new SVM(file);
    double result = i->run();
    // cout << "Average Result = " << result << endl; 
    delete i;
    setlocale(LC_ALL, oldLocale);
    if isnan(result)
       return 100000000000.0; // cout << "WARNING: RESULT IS NAN\n";
    // ONLY FOR REGRESSION 
    if (result < 0.000000000001)
       return 100000000000.0; // cout << "WARNING: RESULT IS NAN\n";
    return result;
}

# endif //NUMPER


