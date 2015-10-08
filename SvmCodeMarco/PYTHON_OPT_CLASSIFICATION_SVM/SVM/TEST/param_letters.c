#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <stdlib.h>
#include "SIMUL.h"
#include "param_letters.h"
#include "utils.h"

#define MAX_STRING_LENGTH 1056

extern SIMULATIONS *insert_simulation(SIMULATIONS *,long int, double);
extern double standard_deviation(double, RESULTS*);

extern word_list line_to_words(char*);
/* extern void free_simulations(SIMULATIONS*);*/

int bili(double x)
{
 if (x == 0) return 0;
 if (x < 0.39) return 1;
 if (x < 0.80) return 2;
 if (x < 1.20) return 3;
 if (x < 2.00) return 4;
 if (x < 3.00) return 5;
 if (x < 4.00) return 6;
else return 7;
}

int alk(double x)
{
 if (x == 0) return 0;
 if (x < 33) return 1;
 if (x < 80) return 2;
 if (x < 120) return 3;
 if (x < 160) return 4;
 if (x < 200) return 5;
 if (x < 250) return 6;
else return 7;
}

int sgot(double x)
{
 if (x == 0) return 0;
 if (x < 13) return 1;
 if (x < 100) return 2;
 if (x < 200) return 3;
 if (x < 300) return 4;
 if (x < 400) return 5;
 if (x < 500) return 6;
else return 7;
}

int albu(double x)
{
 if (x == 0) return 0;
 if (x < 2.1) return 1;
 if (x < 3.0) return 2;
 if (x < 3.8) return 3;
 if (x < 4.5) return 4;
 if (x < 5.0) return 5;
 if (x < 6.0) return 6;
else return 7;
}

int compute_feature(double input, double mean,double sd)
{
double x;

 x = (input - mean) / sd;
 x = x ;
 x = x + 6;

  // printf("x = %d \n",(int) x);

  return (int) x;
}

DATA *postprocess(DATA *data) 
{
SIMULATIONS *sim;
int i;
int dat;
double sd;

  sim = NULL;

  for(i = 0 ; i <  NR_FEATURES ; i++)
    {
     sim = NULL;
     for(dat = 0 ; dat < data[0].tot_data ; dat++)
        sim = insert_simulation(sim,0,data[dat].input[i]);
     sd = standard_deviation(sim->average, sim->results);
     for(dat = 0 ; dat < data[0].tot_data ; dat++)
        {
         data[dat].input[i] =  compute_feature(data[dat].input[i], 
                                 sim->average,sd);
         //printf("feat = %d vale = %d\n",i,data[dat].input[i]);
        }
     //free_simulations(sim);
    }
/*
      sim = NULL;
     for(dat = 0 ; dat < data[0].tot_data ; dat++)
        sim = insert_simulation(sim,0,data[dat].class1);
     sd = standard_deviation(sim->average, sim->results);
     for(dat = 0 ; dat < data[0].tot_data ; dat++)
       {
        data[dat].class =  compute_feature(data[dat].class1, 
                                 sim->average,sd);
         printf("class %d vale = %d\n",dat,data[dat].class);
       }
 */
return data;
}

DATA *read_data(char *filename) {

  FILE *fp;
  char line[MAX_STRING_LENGTH];
  word_list words;
  int nr_pars;
  int found[140],i;
  DATA *data;
  int tot_data;
  double input;
  tot_data = 0;

  data = (DATA*)calloc(28000,sizeof(DATA));
  if (data == NULL)
    {
     printf("Couldn't alocate memory for data\n");
     exit(-1);
    }
  fp = fopen(filename,"r");
  if (fp == NULL) {
    printf("read_control_parameters: Couldn't find file %s...\n",filename);
    exit(-1);
  }
   nr_pars = 0; 
  while (NULL != fgets(line, MAX_STRING_LENGTH, fp))
    if (line[0] != '#') { /* ignore comments */
      words = line_to_words(line);
      if (words.number_of_words != 0) { /* ignore empty lines */
         nr_pars++;
	  

       for(i = 0 ; i < NR_FEATURES ; i++)
	       data[tot_data].input[i] = atof(words.word[i]);
        data[tot_data].class1 = atoi(words.word[NR_FEATURES]);
        tot_data++;
    }
   }
  fclose(fp);
  data[0].tot_data = tot_data;
  data  = postprocess(data);
  return(data);
} /* read_control_parameters */



