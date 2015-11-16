# ifndef NUMPER
# define NUMPER

# include <math.h>
# include <stdio.h>
# include <string.h>
# include <stdlib.h>
# include <vector>
#include<clocale>
#include <float.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>

# include "SIMUL.h"
# include "SVM.h"
# include "param_letters.h"

//# include "SIMUL.h"
//# include "param_letters.h"

#define CLASSES 10
#define BETA_ITER 1
#define AVERAGE_OUT 22.5

#define KERNEL 3
#define LINEAR 1
#define TANH 2
#define GAUSS 3
#define NR_DISTANCES 1

#define TRAINAMOUNT 0.1

extern DATA *read_data(char *);
extern double fabs(double);
extern SIMULATIONS *insert_simulation(SIMULATIONS*, long int, double);
extern void show_simulations(SIMULATIONS *);
extern void write_to_file_all(char*,SIMULATIONS*);

using namespace std;

bool SVM_verbose = false;

double sqr(double x)
{
    return x * x;
}


double SVM::compute_similarity(double *x, double *y, int actionDimension)
{
    if (KERNEL == LINEAR)
     {
      double input = 0.0;
         for (int ad = 0; ad < actionDimension; ++ad)
            input += x[ad] * y[ad];
        return input;
       }
    else if (KERNEL == TANH)
    {
      double input = 0.0;
         for (int ad = 0; ad < actionDimension; ++ad)
            input += x[ad] * y[ad];
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
        double output = exp(-diff / SIGMA);
        return output;
    }

    cout << "*** NO KERNEL SELECTED " << endl;
}

SVM::SVM(const char * pFile)
{
    parameterFile = pFile;
    parameterPath.assign(parameterFile);
    readParameterFile();
}


SVM::~SVM()
{
}
        
void SVM::y_initialize(DATA *leerdata)
{
   for (int episode = 0; episode < leerdata[0].tot_data; ++episode) 
        {
	        int label = leerdata[episode].class1;
                for(int C = 0 ; C < CLASSES ; C++)
                  if (C == label)
                      y[C][episode] = 1.0;
                  else
                      y[C][episode] = -1.0;

        }
}   

void SVM::alpha_initialize(int tot_data, double SVM_C)
{
       for (int C = 0; C < CLASSES; ++C)
            for (int episode = 0; episode < tot_data; ++episode)
	            alpha_coeff[C][episode] =  SVM_C * INIT_ALPHA; // (INIT_ALPHA is between 0 and 1)
}
            
void SVM::compute_kernel_matrix(DATA *leerdata)
{
            for (int episode = 0; episode < leerdata[0].tot_data; ++episode)
	       for (int ad = episode; ad < leerdata[0].tot_data; ++ad)
                  {
                   kernel_train[episode][ad] = compute_similarity(leerdata[episode].input,leerdata[ad].input,NR_FEATURES);
                   kernel_train[ad][episode] = kernel_train[episode][ad];
                  }
}

void SVM::compute_test_kernel_matrix(DATA *leerdata, DATA *testdata)
{
            for (int episode = 0; episode < leerdata[0].tot_data; ++episode)
	       for (int ad = 0; ad < testdata[0].tot_data; ++ad)
                  kernel_test[episode][ad] = compute_similarity(leerdata[episode].input,testdata[ad].input,NR_FEATURES);
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
	
				
               delta_alpha = 1.0 -  sum_1; 
               target = D2 * alpha_coeff[C][episode] + delta_alpha * alpha; 

               if (target > SVM_C)
                     target = SVM_C;
               if (target < 0)
                     target = 0;
               alpha_coeff[C][episode] = target;
            } 
            
            for(int rep_corr1 = 0 ; rep_corr1 < 5 ; rep_corr1++)            
              for (int C = 0; C < CLASSES; ++C)
              { 
                  double sum_2 = 0.0;
                  for(int i = 0 ; i < tot_data ; i++)
                     sum_2 += alpha_coeff[C][i] * y[C][i];
                  
               for (int episode = 0; episode < tot_data; ++episode)
              {
                     double old_val = alpha_coeff[C][episode];
                     delta_alpha = - 2 * COST * sum_2 * y[C][episode];
                     target = alpha_coeff[C][episode] + delta_alpha * alpha; 
                     if (target > SVM_C)
                        target = SVM_C;
                     if (target < 0)
                        target = 0;
                     alpha_coeff[C][episode] = target;
                     sum_2 += (alpha_coeff[C][episode] - old_val) * y[C][episode];
                  }
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
                                bias[C] += y[C][i] - output_SVM[C]; 
	                               tot++;
                                 /*
                                     if ((y[C][i] * output_SVM[C]) < 0.0)
                                      {
                                         bias[C] += y[C][i] - output_SVM[C];
	                               tot++;
                                     }
                                 */
	                    }
                    }
                    if (tot == 0)
                     bias[C] = 0.0;
                   else
                      bias[C] /= tot;
                } 
}
                


/* TEST op trainset */ 
double SVM::compute_train_error(DATA *train_example, int first_id, int tot_data)
{
   int label = train_example[0].class1;
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

        for ( int C = 1 ; C < CLASSES ; C++ ) {
            if ( rewardsPerClass[C] > maxRewardPerClass ) {
                maxRewardPerClass = rewardsPerClass[C] ;
                nMax = 1 ;
            } else if ( rewardsPerClass[C] == maxRewardPerClass ) {
                nMax++ ;
            }
        }
        if ( rewardsPerClass[ label ] == maxRewardPerClass ) {
            TrainResult = 1.0 - 1.0/nMax ;
        } else {
            TrainResult= 1.0 ;
        }
return TrainResult;
}
	            
double SVM::compute_test_error(DATA *testdata, int episode2, DATA *leerdata)
{
   double F_episode;
   double TrainResult;
   double *rewardsPerClass    = new double[CLASSES];

   int label = testdata[episode2].class1;
   for (int C = 0; C < CLASSES; ++C)
        {
          output_SVM[C] = 0.0;
          for (int j = 0; j < leerdata[0].tot_data; ++j)
             {
              F_episode = kernel_test[j][episode2];
              output_SVM[C] += alpha_coeff[C][j] * y[C][j] * F_episode;
             }
            output_SVM[C] += bias[C];
          } 
      for (int C = 0; C < CLASSES; ++C)
        rewardsPerClass[C] = output_SVM[C];
            
        double maxRewardPerClass = rewardsPerClass[0] ;
        int nMax = 1 ;

        for ( int C = 1 ; C < CLASSES ; C++ ) {
            if ( rewardsPerClass[C] > maxRewardPerClass ) {
                maxRewardPerClass = rewardsPerClass[C] ;
                nMax = 1 ;
            } else if ( rewardsPerClass[C] == maxRewardPerClass ) {
                nMax++ ;
            }
        }
        if ( rewardsPerClass[ label ] == maxRewardPerClass ) {
            TrainResult = 1.0 - 1.0/nMax ;
        } else {
            TrainResult= 1.0 ;
        }
return TrainResult;
}

double SVM::run()
{
    double AverageresultPerEpisode = runExperiment();
	
    return AverageresultPerEpisode;
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
            
    bias = new double[CLASSES];
    output_SVM = new double[CLASSES];

    data = NULL;
    double best_train;
    double best_test;
    int best_found;


    data = read_data("Bangla_digit_36pixels");
    size_t num_data = data[0].tot_data + 1;
    DATA *leerdata = new DATA[num_data];
    DATA *testdata = new DATA[num_data];
    y = new double*[CLASSES];
    alpha_coeff = new double*[CLASSES];

    for (size_t C = 0; C < CLASSES; ++C)
    {
        y[C] = new double[num_data];
        alpha_coeff[C] = new double[num_data];
    }
    kernel_train = (double **) calloc(TRAINAMOUNT * num_data,sizeof(double*));
     for(int i = 0 ; i < TRAINAMOUNT * num_data ; i++)
        kernel_train[i] = (double *) calloc(TRAINAMOUNT * num_data,sizeof(double));
    kernel_test = (double **) calloc(TRAINAMOUNT * num_data,sizeof(double*));
     for(int i = 0 ; i < TRAINAMOUNT * num_data ; i++)
        kernel_test[i] = (double *) calloc((1 - TRAINAMOUNT) * num_data,sizeof(double));

    for (int rep = 0; rep < NR_REP1; ++rep)
    {
      for (int C = 0; C < CLASSES; ++C)
              bias[C] = 0;
    if (data != NULL)
        free(data);
    data = read_data("Bangla_digit_36pixels");
        best_train = DBL_MAX;
            leerdata[0].tot_data = 0;
            testdata[0].tot_data = 0;
            for (i = 0; i < data[0].tot_data; ++i)
            {
                if (leerdata[0].tot_data >= (data[0].tot_data * TRAINAMOUNT))
                {
                    testdata[testdata[0].tot_data].class1 = data[i].class1;
                    for (j = 0; j < NR_FEATURES; ++j)
                        testdata[testdata[0].tot_data].input[j] = data[i].input[j];
                    ++testdata[0].tot_data;
                }
                else if (testdata[0].tot_data >= (data[0].tot_data * (1.0 - TRAINAMOUNT)))
                {
                    leerdata[leerdata[0].tot_data].class1 = data[i].class1;
                    for (j = 0; j < NR_FEATURES; ++j)
                        leerdata[leerdata[0].tot_data].input[j] = data[i].input[j];
                    ++leerdata[0].tot_data;
                }
                else if (drand48() < ((1 - TRAINAMOUNT)))
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
    if (data != NULL)
        free(data);
     data = NULL;

   // INITIALIZE
  
   y_initialize(&leerdata[0]);
   alpha_initialize(leerdata[0].tot_data,SVM_C);
   
   compute_kernel_matrix(leerdata); 
   compute_test_kernel_matrix(leerdata,testdata); 


        for (int rep2 = 0; rep2 <= NR_REP2; ++rep2)
        {
          train(leerdata[0].tot_data);
          compute_bias(leerdata[0].tot_data); 
                
           double averageRewards = 0;
           for (int e = 0; e < leerdata[0].tot_data; ++e)
	        averageRewards += compute_train_error(&leerdata[e], e, leerdata[0].tot_data);
	
           averageRewards /= leerdata[0].tot_data;
 
           SIMULTRAIN = insert_simulation(SIMULTRAIN, rep2, averageRewards);
           show_simulations(SIMULTRAIN);
           if (averageRewards <= best_train) 
                { 
                    best_train = averageRewards;
                    best_found = 1;
                }
           else 
                 best_found = 0;
                 best_found = 1; // BEST AT END 

           averageRewards = 0;
           for (int e = 0; e < testdata[0].tot_data; ++e)
	          averageRewards += compute_test_error(&testdata[0],e, &leerdata[0]);
	
           averageRewards /= testdata[0].tot_data;
  
           SIMULTEST = insert_simulation(SIMULTEST, rep2, averageRewards);
           show_simulations(SIMULTEST);
           if (best_found)
                 best_test = averageRewards;
            }
        SIMULBEST = insert_simulation(SIMULBEST, 0, best_test);
        show_simulations(SIMULTRAIN);
        show_simulations(SIMULTEST);
        show_simulations(SIMULBEST);
        sprintf(fileout,"Results/");
        strcat(fileout,"ALEX_SVM");
        sprintf(added,"_alpha:%1.11f_beta:%1.7f_COST:%1.3f_D2:%1.1f_SVMC:%1.1f_ALPHAITER:%d_REP1:%d_REP2:%d_EPS:%1.2f",
				alpha,beta,COST,D2,SVM_C,ALPHA_ITER,NR_REP1,NR_REP2,EPS);
        strcat(fileout,added);
//        printf("%s\n",fileout);
         //write_to_file_all(fileout,SIMULTEST);
        strcat(fileout,"_BEST");
         //write_to_file_all(fileout,SIMULBEST);
       
    }

    delete [] leerdata;
    delete [] testdata;
    delete [] data;

 /*
    for (size_t C = 0; C < CLASSES; ++C)
    {
        delete [] y[C];
        delete [] alpha_coeff[C];
    }

    delete [] rewardsPerClass;
  */
    return SIMULBEST->average;
}

void SVM::readParameterFile()
{
    ifstream ifile;
    
    ifile.open(parameterFile, ifstream::in);
    
    string temp;
    
    while (temp.compare( "algorithm" ) != 0)
	    ifile >> temp;
	 
	ifile >> temp >> alpha;
	ifile >> temp >> beta;
	ifile >> temp >> COST;
	ifile >> temp >> D2;
	ifile >> temp >> SVM_C;
	ifile >> temp >> ALPHA_ITER;
	ifile >> temp >> NR_REP1;
	ifile >> temp >> NR_REP2;
	ifile >> temp >> EPS;
	ifile >> temp >> SIGMA;
	ifile >> temp >> INIT_ALPHA;
	 
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
    if isnan(result)
        cout << "WARNING: RESULT IS NAN\n";
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
    delete i;
    setlocale(LC_ALL, oldLocale);
    if isnan(result)
        cout << "WARNING: RESULT IS NAN\n";
    return result;
}

# endif //NUMPER


