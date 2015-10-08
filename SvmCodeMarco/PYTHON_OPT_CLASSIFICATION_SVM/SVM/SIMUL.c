#include <stdio.h>
#include <math.h>
#include <string.h>
#include <cstdlib>
#include <stdlib.h>
#include "SIMUL.h"

#include <vector>

std::vector<RESULTS *> alloced_results;
std::vector<SIMULATIONS *> alloced_simulations;

void free_simulations()
{
    for (std::vector<RESULTS *>::iterator i = alloced_results.begin(); i != alloced_results.end(); ++i)
        free(*i);
    alloced_results.clear();
    for (std::vector<SIMULATIONS *>::iterator i = alloced_simulations.begin(); i != alloced_simulations.end(); ++i)
        free(*i);
    alloced_simulations.clear();
}

RESULTS *new_results(double performance)
{
RESULTS *results;

  results = (RESULTS *) malloc(sizeof(RESULTS));
  if (results == NULL)
    {
      printf("Couldn't allocate memory for results ! \n");
      exit(0);
    }
  alloced_results.push_back(results);
  results->rest = NULL;
  results->last = results;
  results->performance = performance;
  return results;
 }

RESULTS *insert_result(double performance, RESULTS *results)
{
 RESULTS *new_res;

  new_res = new_results(performance);
   if (results == NULL)
      return new_res;
   else
      {
       results->last->rest = new_res;
       results->last = new_res;
      }
 
   return results;
}

double standard_deviation(double average,RESULTS *results)
{
double variance=0.0;
RESULTS *res;
int total=0;

  if (results == NULL) return 0.0;
    for(res = results ; res != NULL ; res = res->rest)
       {
        variance += (res->performance - average) *
           	   (res->performance - average);
        total++;
       }
  variance = variance / total;
  return sqrt(variance);
}

void show_simulations(SIMULATIONS *simul)
{
SIMULATIONS *p;

   /* PUT RETURN ON FOR PSO RUNS */
    //return;
  printf("\nNR_MAZES   NR_SIMUL  averagee         S.D.   MAX     MIN  \n");
  for(p = simul; p!= NULL; p=p->rest)
      {
       p->standard_dev = standard_deviation(p->average,p->results);
       printf("%ld         %d        %f     %f   %f   %f\n",
	 p->nr_games,p->nr_simulations,p->average,p->standard_dev,p->max,
							 p->min);
      }
  printf("\n");
}

SIMULATIONS *new_simulations()
{  
SIMULATIONS *simulations;

   simulations = (SIMULATIONS *) malloc(sizeof(SIMULATIONS));
   if (simulations == NULL)
    {
      printf("Couldn't allocate memory for results ! \n");
      exit(0);
    } 
   alloced_simulations.push_back(simulations);
   simulations->nr_simulations = 0;
   simulations->rest = NULL;
   simulations->total = 0.0;
   simulations->max = -1.0;
   simulations->min = 9999999.0;
   simulations->standard_dev = 0.0;
   simulations->results = NULL;
   simulations->last_active = simulations;
   return simulations;
}

SIMULATIONS *insert_simulation(SIMULATIONS *simulations,
       long int nr_games, double performance)
{ 
SIMULATIONS *simul,*last_sim, *simul_before, *simul_after, *search_point;

  last_sim = NULL;
  if ((simulations == NULL) || (simulations->last_active->nr_games > nr_games))
    search_point = simulations;
  else
     search_point = simulations->last_active;
 
 for(simul = search_point; ((simul!=NULL) && (simul->nr_games < nr_games));
			   simul = simul->rest)
      last_sim = simul;

 if (simul == NULL)
 {
  simul = new_simulations();
  simul->nr_simulations++;
  simul->nr_games = nr_games;
  simul->total += performance;
  simul->max = performance;
  simul->min = performance;
  simul->average = performance;
  simul->results = new_results(performance);
  if (simulations == NULL)
	return simul;
   else
     last_sim->rest = simul;
  simulations->last_active = simul;
  }
  else if (simul->nr_games == nr_games)
    {
     simul->total += performance;
     simul->nr_simulations++;
     simul->average = simul->total/simul->nr_simulations;
     simul->results = insert_result(performance, simul->results);
      if (performance > simul->max)
          simul->max = performance;
      if (performance < simul->min)
          simul->min = performance;
  simulations->last_active = simul;
   }
 else 
   {
  simul_before = last_sim;
  simul_after = simul;
  simul = new_simulations();
  simul->nr_simulations++;
  simul->nr_games = nr_games;
  simul->total += performance;
  simul->max = performance;
  simul->min = performance;
  simul->average = performance;
  simul->results = new_results(performance);
  simul->rest = simul_after;
  if (simul_before == NULL)
     return simul;
  else     
     simul_before->rest = simul;
  simulations->last_active = simul;
  }
return simulations;
}

void write_to_file(char file_name[20],SIMULATIONS *sim1)
{
FILE *fp;
SIMULATIONS *p;

  fp = fopen(file_name,"w");
  if (fp == NULL)
     {
      printf("Could not open file_name \n");
      exit(0);
     }

 fprintf(fp, "NR MAZES NR SIM average SD MAX MIN \n");
 for(p = sim1; p!= NULL ; p=p->rest)
   fprintf(fp,"%ld %f \n",
     p->nr_games,p->average);
  fprintf(fp,"\n \n");

  fclose(fp);
}

void write_to_file_sd(char file_name[20],SIMULATIONS *sim1)
{
FILE *fp;
SIMULATIONS *p;

  fp = fopen(file_name,"w");
  if (fp == NULL)
     {
      printf("Could not open file_name \n");
      exit(0);
     }

 fprintf(fp, "NR MAZES NR SIM average SD MAX MIN \n");
 for(p = sim1; p!= NULL ; p=p->rest)
    {
     p->standard_dev = standard_deviation(p->average,p->results);
     fprintf(fp,"%ld %f %f\n", p->nr_games,p->average,p->standard_dev);
    }
    fprintf(fp,"\n \n");

  fclose(fp);
}


void write_to_file_all(char file_name[20], SIMULATIONS *sim1)
{
FILE *fp;
SIMULATIONS *p;

  fp = fopen(file_name,"w");
  if (fp == NULL)
     {
      printf("Could not open file_name \n");
      exit(0);
     }

 fprintf(fp, "NR EPOCHS NR SIM average SD MAX MIN (epoch -> Exploration)\n");
 for(p = sim1; p!= NULL ; p=p->rest)
    {
     p->standard_dev = standard_deviation(p->average,p->results);
   fprintf(fp,"%ld %d %f %f %f %f\n",
     p->nr_games,p->nr_simulations,p->average,p->standard_dev,p->max,
     p->min);
    }
  fprintf(fp,"\n \n");
  
  fclose(fp);
}
