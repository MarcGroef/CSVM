import random
import os
from numpy import array, zeros, ones, fabs, sqrt, log, argmin, argsort, log2, random
from param_generator import ParameterGenerator
import cma

generators = ['CMAES_advanced']

class CMAES_advanced(ParameterGenerator):
    """
    The CMAES_advanced subclass of ParameterGenerator
    returns the next 
    """
    def __init__(self):
        """The constructor initializes empty lists"""
        super(CMAES_advanced, self).__init__()

        self.counter = 0
        self.best = None

        self.fitnesses = []
        self.solutions = []
        self.available = []
        self.eval_order = []
        self.averages = []
        self.reps = []
        self.ucb = [] 
        self.sendout = []
        self.sendout_old = []
        self.minimum = []
        self.maximum = []

        self.best_params = None
        self.pcmaes = None
        self.first_gen = True
        self.done = False

    def psetup_cmaes(self):
        xstart = []
        
        print "Maximum number of repetitions: %d" % self.max_reps
        print "Config: %s" % repr(self.param_settings)
        for idx in range(len(self.param_names)):
            settings = self.param_settings[idx]
            name = self.param_names[idx]
            if settings['type'] == "static":
                continue
            self.minimum.append(self.param_settings[idx]['min'])
            self.maximum.append(self.param_settings[idx]['max'])
            if settings['scaling'] == "linear":
                print "Linear scaling for parameter %s. \n\
                       Shown are the bounds between which values have the same sensitivity " %name
                bounds =[(self.reverse_linear_scaling(self.minimum[-1],
                                                      self.maximum[-1],
                                                      x )) for x in range(11)] # because scaling is from 0 to 10
                print [("%.2f" %x) for x in bounds]
            elif settings['scaling'] == "log":
                print "Log scaling for parameter %s. \n\
                       Shown are the bounds between which values have the same sensitivity " %name
                bounds =  [(self.reverse_log_scaling(self.minimum[-1],
                                                     self.maximum[-1],
                                                     x )) for x in range(11)]
                print [("%.2e" %x) for x in bounds]

              

        print "Number of dimensions: %d" % len(self.minimum)
        xstart = ones(len(self.minimum))  # just a dummy, not used as first generation is sampled uniformly
        self.pcmaes = cma.CMAEvolutionStrategy(xstart, 3, {'bounds':[0, 10]}) # because everything is scaled to [0,10] anyways


    def set_result(self, params, result, id=0):
        """
        This method sets the result/evalution of the last set of
        parameters. This can be used by subclasses to generate the next
        value for each parameter.
        """
        # Double results - ignore
        if id not in self.sendout:
            return

        self.sendout.pop(self.sendout.index(id)) # register that result for send out offspring was returned
        self.reps[id] += 1
        self.counter += 1
        self.averages[id] = self.averages[id] + (1.0/self.reps[id]) * (result - self.averages[id]) 
                            # incremental calculation of averages
        corrected_result = None
        if self.generation_done():
            gen_best = min(self.averages)
            gen_best_idx = argmin(self.averages)
            gen_best_params = self.solutions[gen_best_idx]
            corrected_result = (gen_best_params, gen_best)
            if self.best is None or gen_best < self.best:
                self.best = gen_best
                self.best_params = gen_best_params
                print ""
                print ""
                print "================================================================================"
                print "================================================================================"
                print "new best found " + repr(self.best) + " " + repr(self.to_outside_repr(self.best_params)) 
                print "================================================================================"
                print "================================================================================"
                print ""
                print ""

            self.pcmaes.tell(self.solutions, self.averages)
        return corrected_result

    def get_best_result(self):
        if self.best is None:
            best = min(self.averages)
            best_idx = argmin(self.averages)
            best_params = self.solutions[best_idx]
        else:
            best = self.best
            best_params = self.best_params
        if best_params is None:
            return None, None
        vector = [(x[1]) for x in self.to_outside_repr(best_params)]
        best_params = {}
        for idx in range(len(self.param_names)):
            name = self.param_names[idx]
            settings = self.param_settings[idx]
            value = vector[idx]
            if settings['type'] == "int":
                value = int(round(value))
            elif settings['type'] == "float":
                value = float(value)
            best_params[name] = value
        return best_params, best

    def get_new_set(self):
        """get new generation of solutions"""
        self.fitnesses = []
        self.averages = []
        self.param_values = []
        self.counter  = 0
        if self.first_gen: # pick first gen uniformly distributed
            print "first generation is chosen uniformly distributed in search space"
            self.pcmaes.ask() # not used, only used  to make the ask and tell interface believe that there has already 
                              # been an iteration, so that it accepts function values back
            self.solutions = [(random.uniform(0,10,len(self.minimum))) for _ in range(self.pcmaes.popsize)]
            self.first_gen = False
        else:
            self.solutions = self.pcmaes.ask()

        self.eval_order = range(self.pcmaes.popsize) # evaluatate every new offspring once as init for ucb
        self.reps = zeros(self.pcmaes.popsize)
        for i in enumerate(self.solutions):
            self.fitnesses.append(None)
            self.averages.append(0)
            self.param_values.append([])

    def generation_done(self):
        """Check if the current generation finished evaluating"""
        if self.ucb_init_done() and not self.sendout:
            return True
        else:
            print "Still waiting for %s" % repr(self.sendout)
            return False

    def reached_max_reps(self):
        """Check if at least one of the offspring is evaluated the required number of times"""
        if len(self.reps) == 0:
           return True
        return max(self.reps) >= self.max_reps  

    def ucb_init_done(self):
        """Check if the intial evaluation run of ucb is done """
        done = True
        for elem in self.reps:
            if elem < 1:
                return False
        return done 


    def generate_parameters(self, id=0):
        """
        This method returns the currently most promising parameter
        to be evaluated again.
        """
        if not self.pcmaes:
            self.psetup_cmaes()

        if self.reached_max_reps() and self.generation_done():
            self.get_new_set()
            print "New Generation"
        elif self.reached_max_reps():
            return 0, None   # wait for all sendout parameters to return
         
        if len(self.eval_order) == 0:
            self.bandit()
            return 0, None   # Nothing to process yet
        
        if self.ucb_init_done() and not self.reached_max_reps():
            self.bandit()

        if len(self.sendout) >= self.pcmaes.popsize:
            return 0, None # still busy with the rest
        
        k = self.eval_order.pop(0)
        self.sendout.append(k)
        #print "next offspring to be evaluated is %d" % k
        pvector = self.solutions[k]
        new_value = [(x[1]) for x in self.to_outside_repr(pvector)]
        #print new_value 
        self.param_values[k].extend(new_value)
        return k, self.param_values[k]


    def get_next_parameter(self, name, value, settings, id=0):
        pass

    def bandit(self):
      """
      for each generation, manages in which order and how often 
      each offspring is evaluated.
      offspring that are not promising will not be further evaluated.
      uses Csaba Szepesvri's ucb1 (upper confidence bounds)
      """
      self.ucb = [(x - (sqrt((2* log(self.counter))/n))) for x, n  in zip(self.averages, self.reps)]
      if self.ucb_init_done():
          if self.eval_order == list(argsort(self.ucb)): # if nothing changed
              if self.sendout_old == self.sendout :
                  self.eval_order = list(argsort(self.ucb))[0:2] 
              else:
                  self.eval_order = list(argsort(self.ucb)) 
          else:
              self.eval_order = list(argsort(self.ucb)) 
          self.sendout_old = self.sendout
      else:
          self.eval_order.extend(list(argsort(self.ucb)))
          self.eval_order = self.eval_order[::self.pcmaes.popsize]
      
      
          
          
    def reverse_linear_scaling(self,minimum,maximum,y):
       """
       parameters from linear scaling back to normal 
       so they can be handed over to be evaluated
       """
       value = minimum + y * (maximum-minimum)/10.0
       return value
    
    def reverse_log_scaling(self,minimum,maximum,y):
       """
       parameters from logspace back to normal 
       so they can be handed over to be evaluated
       """
       value = minimum * (maximum/minimum)**(y/10.0)
       return value
    
    def to_outside_repr(self, pvector):
       """
       converts scaled output back to the actual parameter space
       """
       vector_counter = 0
       out = []
       for idx  in range(len(self.param_names)):
           name = self.param_names[idx]
           settings = self.param_settings[idx]
           if settings['type'] == "static":
               new_value = settings['value'];
           elif settings['scaling'] == "linear":
               new_value = self.reverse_linear_scaling(self.minimum[vector_counter],
                                                       self.maximum[vector_counter],
                                                       pvector[vector_counter])
               if settings['type'] ==  "int":
                   new_value = int(round(new_value))
               vector_counter += 1
           elif settings['scaling'] == "log":
               new_value = self.reverse_log_scaling(self.minimum[vector_counter],
                                                    self.maximum[vector_counter],
                                                    pvector[vector_counter])
               if settings['type'] ==  "int":
                   new_value = int(round(new_value))
               vector_counter += 1
           out.append((name, new_value))
       return out 
