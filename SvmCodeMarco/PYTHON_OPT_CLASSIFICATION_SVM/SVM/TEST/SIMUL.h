#ifndef SIMUL
#define SIMUL

typedef struct results_struct RESULTS;
struct results_struct {
       
      RESULTS *last;
      RESULTS *rest;
      double performance;
  };


typedef struct simul_struct SIMULATIONS;
struct simul_struct {
       SIMULATIONS *rest;
       SIMULATIONS *last_active;
       long int nr_games;
       double max;
       double min;
       double average;
       double total;
       double standard_dev;
       int nr_simulations;
       RESULTS *results;
       };
#endif //SIMUL
