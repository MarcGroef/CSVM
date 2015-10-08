#include <stdio.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <stdlib.h>
#include "SIMUL.h"
#include "param_letters.h"
#include "utils.h"

#define MAX_STRING_LENGTH 425600

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
  int i;
  DATA *data, *hulpdata;
  int tot_data;
  double input;
  double max[50380],min[50380];
  int used[12000];
  int NR_FEATURES1 = 784;
 
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
int classes1[10]; 
    for(i = 0 ; i < 10; i++)
          classes1[i] = 0;
   nr_pars = 0; 
  while (NULL != fgets(line, MAX_STRING_LENGTH, fp))
    if (line[0] != '#') { /* ignore comments */
      words = line_to_words(line);
      if (words.number_of_words != 0) { /* ignore empty lines */
         nr_pars++;
	 

       for(i = 0 ; i < NR_FEATURES1 ; i++)
	   {
     	      data[tot_data].input[i] = atof(words.word[i]);
     	      /* if ((data[tot_data].input[i] != -1.0)
     	       && (data[tot_data].input[i] != 0.0)
     	       &&(data[tot_data].input[i] != 1.0))
                   printf("Data %f \n", data[tot_data].input[i]);*/

               if (data[tot_data].input[i] > max[i])
                   max[i] = data[tot_data].input[i];
                 if (data[tot_data].input[i] < min[i])
                   min[i] = data[tot_data].input[i];
          }
     	   data[tot_data].class1 = atoi(words.word[NR_FEATURES1]) - 1;
           classes1[data[tot_data].class1]++;
	 tot_data++;
    }



    for(i = 0 ; i < words.number_of_words ; i++)
       free(words.word[i]);
    free(words.word);
   }
  fclose(fp);
  data[0].tot_data = tot_data;
  for(int dat = 0 ; dat < data[0].tot_data ; dat++)
       for(i = 0 ; i < NR_FEATURES1 ; i++)
           {
               data[dat].input[i] = 1.0 - (data[dat].input[i] / 255.0);
           }

  double ****digits;
  digits = (double ****)calloc(11000,sizeof(double***));
  for(int i = 0 ; i < 11000 ; i++)
  digits[i] = (double ***)calloc(28,sizeof(double**));
  for(int i = 0 ; i < 11000 ; i++)
   for(int j = 0 ; j < 28 ; j++)
  digits[i][j] = (double **)calloc(28,sizeof(double*));
  for(int i = 0 ; i < 11000 ; i++)
   for(int j = 0 ; j < 28 ; j++)
   for(int l = 0 ; l < 28 ; l++)
  digits[i][j][l] = (double *)calloc(9,sizeof(double));
   

    for(int dat = 0 ; dat < data[0].tot_data ; dat++)
       for(i = 0 ; i < NR_FEATURES1 ; i++)
          digits[dat][i / (28)][i % (28)][8] = data[dat].input[i];

   int ind;
   int off_x, off_y;
   int off_x2, off_y2;
   for(int dat = 0 ; dat < data[0].tot_data ; dat++)
      for(int x = 0 ; x < 28 ; x++)
       for(int y = 0 ; y < 28 ; y++)
        for(int dir = 0 ; dir < 4 ; dir++)
         for(int len = 0 ; len < MAX_LEN ; len++)
          {
            ind = len + dir * MAX_LEN +  y * 4 * MAX_LEN + x * 28 *  4 * MAX_LEN;
            if (dir == 0)
               {
                 off_x = -1;
                 off_y = -1; 
               }
            if (dir == 1)
               {
                 off_x = 0;
                 off_y = -1; 
               }
            if (dir == 2)
               {
                 off_x = 1; 
                off_y = -1; 
               }
            if (dir == 3)
               {
                 off_x = 1;
                 off_y = 0; 
               }
             double other_value = 0.0;
             double other_value2;
             for(int len_1 = -len - 1 ; len_1 < len + 1 ; len_1++)
               {
                  off_x2 = off_x * len_1;
                  off_y2 = off_y * len_1;
                  if (((x - off_x2) < 0) || ((x - off_x2) > 27) || ((y - off_y2) < 0) ||( (y - off_y2) > 27))
                      other_value += 0.0;
                   else
                       other_value += digits[dat][x - off_x2][y - off_y2][8];
               } 
              data[dat].input[ind] = other_value; // / sqrt(2 + len * 2 + 1);
            }
   for(int dat = 0 ; dat < data[0].tot_data ; dat++)
       for(i = 0 ; i < NR_FEATURES1 ; i++)
        data[dat].input[i + ind]= digits[dat][i / (28)][i % (28)][8];

    for(i = 0 ; i < 10; i++)
         printf("CLASS %d nr elt: %d\n",i,classes1[i]);

/*
     for(i = 0 ; i < NR_FEATURES ; i++)
          {
           max[i] = -100000.0;
           min[i] = 100000.0;
          }

  for(int dat = 0 ; dat < data[0].tot_data ; dat++)
       for(i = 0 ; i < NR_FEATURES ; i++)
	   {
               if (data[dat].input[i] > max[i])
                   max[i] = data[dat].input[i];
                 if (data[dat].input[i] < min[i])
                   min[i] = data[dat].input[i];
          }
  for(int dat = 0 ; dat < data[0].tot_data ; dat++)
       for(i = 0 ; i < NR_FEATURES ; i++)
           if (max[i] == min[i])
              data[dat].input[i] = 0.0;
           else
            data[dat].input[i] = 2.0 * ((data[dat].input[i] - min[i]) / (max[i] - min[i])) - 1.0;
*/

/*
  for(int dat = 0 ; dat < data[0].tot_data ; dat++)
       for(i = 0 ; i < NR_FEATURES ; i++)
           {
            if (max[i] == 0.0)
              data[dat].input[i] = 0.0;
             else
               data[dat].input[i] = (data[dat].input[i] / max[i]);
           }
*/
/*
  for(int dat = 0 ; dat < data[0].tot_data ; dat++)
       for(i = 0 ; i < NR_FEATURES ; i++)
           {
            if (max[i] == min[i])
              data[dat].input[i] = 0.0;
             else
               data[dat].input[i] = 2.0 * ((data[dat].input[i] - min[i]) / (max[i] - min[i])) - 1.0 ;
//            cout << "feat " << i << " value = " << data[dat].input[i] << endl;
           }
*/
/*
  for(int dat = 0 ; dat < 1500; dat++)
    {
      double dif = 0.0;
      for(i = 0 ; i < NR_FEATURES ; i++)
          dif += (data[dat].input[i] - data[0].input[i]) * (data[dat].input[i] - data[0].input[i]);
      printf("DIF to %d = %f\n",dat,dif);
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



