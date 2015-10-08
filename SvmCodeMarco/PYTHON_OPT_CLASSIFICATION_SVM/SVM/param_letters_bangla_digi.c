#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <stdlib.h>
#include "SIMUL.h"
#include "param_letters.h"
#include "utils.h"

#define MAX_STRING_LENGTH 32560

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
 /*x = x ;
 x = x + 4;
 */
  // printf("x = %d \n",(int) x);

  return  x;
  // return (int) x;
}

DATA *read_data(char *filename) {

  FILE *fp;
  char line[MAX_STRING_LENGTH];
  word_list words;
  int nr_pars;
  int found[5380],i;
  DATA *data, *hulpdata;
  int tot_data;
  double input;
  double max[5380],min[5380];
  int used[9000];
 
   tot_data = 0;

       for(i = 0 ; i < NR_FEATURES ; i++)
          {
           max[i] = -100000.0;
           min[i] = 100000.0;
          }

  data = (DATA*)calloc(12000,sizeof(DATA));
  hulpdata = (DATA*)calloc(12000,sizeof(DATA));
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
	   {
     	      data[tot_data].input[i] = atoi(words.word[i]);
               if (data[tot_data].input[i] > max[i])
                   max[i] = data[tot_data].input[i];
                 if (data[tot_data].input[i] < min[i])
                   min[i] = data[tot_data].input[i];
          }
     	   data[tot_data].class1 = atoi(words.word[NR_FEATURES]) - 1;
	 tot_data++;
    }
    for(i = 0 ; i < words.number_of_words ; i++)
       free(words.word[i]);
    free(words.word);
   }
  fclose(fp);
  data[0].tot_data = tot_data;
  for(int dat = 0 ; dat < data[0].tot_data ; dat++)
       for(i = 0 ; i < NR_FEATURES ; i++)
           {
            if (max[i] == 0.0)
              data[dat].input[i] = 0.0;
             else
               data[dat].input[i] = (data[dat].input[i] / max[i]);
//            cout << "feat " << i << " value = " << data[dat].input[i] << endl;
           }

/*  for(int dat = 0 ; dat < data[0].tot_data ; dat++)
       for(i = 0 ; i < NR_FEATURES ; i++)
           {
            if (max[i] == min[i])
              data[dat].input[i] = 0.0;
             else
               data[dat].input[i] = 2.0 * ((data[dat].input[i] - min[i]) / (max[i] - min[i])) - 1.0 ;
//            cout << "feat " << i << " value = " << data[dat].input[i] << endl;
           }
*/

  for(int dat = 0 ; dat < data[0].tot_data ; dat++)
     used[dat] = 0;
  int use = 0;
  int select;
  while(use < data[0].tot_data)
   {
    select = drand48() * (data[0].tot_data);
    if (used[select] == 0) 
     {
      used[select] = 1;
       for(i = 0 ; i < NR_FEATURES ; i++)
         hulpdata[use].input[i] = data[select].input[i];
       hulpdata[use].class1 = data[select].class1;
       use++;
     }
   }
  hulpdata[0].tot_data = data[0].tot_data;


  free(data);
  return(hulpdata);
} /* read_control_parameters */



