import random
import os
from numpy import array, zeros, ones, fabs, sqrt, log, argmin, argsort, log2, random, var,argmax, log10
import numpy
from param_generator import ParameterGenerator
import cma

generators = ['Random_bandit']

class Random_bandit(ParameterGenerator):
    """
    The Random_bandit subclass of ParameterGenerator
    returns the next 
    """
    def __init__(self):
        """The constructor initializes empty lists"""
        super(Random_bandit, self).__init__()

        self.counter = 0
        self.best = None

        self.fitnesses = []
        self.solutions = []
        self.available = []
        self.eval_order = []
        self.averages = []
        self.rewards_of_plays = []
        self.reps = []
        self.ucb = [] 
        self.sendout = []
        self.sendout_old = []
        self.minimum = []
        self.maximum = []
        self.dim = None

        self.best_params = None
        self.random_setup = False
        self.done = False

    def psetup_random(self):
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
        self.dim = len(self.minimum)
        self.random_popsize = 8 
        self.rewards_of_plays = [([]) for _ in range(self.random_popsize)]
        self.random_setup = True


    def set_result(self, params, result, id=0):
        """
        This method sets the result/evalution of the last set of
        parameters. This can be used by subclasses to generate the next
        value for each parameter.
        """
        print '%s got returned' %id
        # Double results - ignore
        if id not in self.sendout:
            print '%s returned but not in sendout anymore' %id
            return

        self.sendout.pop(self.sendout.index(id)) # register that result for send out offspring was returned
        self.reps[id] += 1
        self.counter += 1
        self.averages[id] = self.averages[id] + (1.0/self.reps[id]) * (result - self.averages[id]) 
                            # incremental calculation of averages
        self.rewards_of_plays[id].append(result)
        corrected_result = None
        if self.generation_done():
            print "generation  done!"
            print zip(self.reps, self.averages, self.ucb)

            #gen_best = min(self.averages)
            #gen_best_idx = argmin(self.averages)
            #gen_best_params = self.solutions[gen_best_idx]
            #corrected_result = (gen_best_params, gen_best)

            # changes for confidence levels
            gen_best_idx = argmax(self.reps)
            gen_best = self.averages[gen_best_idx]
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

        return corrected_result

    def get_best_result(self):
        if self.best is None:
            gen_best_idx = argmax(self.reps)
            best = self.averages[gen_best_idx]
            best_params = self.solutions[gen_best_idx]
            #best = min(self.averages)
            #best_idx = argmin(self.averages)
            #best_params = self.solutions[best_idx]
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
        self.rewards_of_plays = [([]) for _ in range(self.random_popsize)]
        self.counter  = 0
        self.solutions=random.ranf((self.random_popsize,  self.dim))*10

        self.eval_order = range(self.random_popsize) # evaluatate every new offspring once as init for ucb
        self.reps = zeros(self.random_popsize)
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
        print 'self.reps'
        print self.reps
        for elem in self.reps:
            if elem < 1:
                return False
        return done 


    def generate_parameters(self, id=0):
        """
        This method returns the currently most promising parameter
        to be evaluated again.
        """
        if not self.random_setup:
            self.psetup_random()

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

        if len(self.sendout) >= self.random_popsize:
            return 0, None # still busy with the rest
        
        k = self.eval_order.pop(0)
        #print "next offspring to be evaluated is %d" % k
        pvector = self.solutions[k]
        new_value = [(x[1]) for x in self.to_outside_repr(pvector)]
        #print new_value 
        self.param_values[k].extend(new_value)
        self.sendout.append(k)
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
      #self.ucb = [(x - (sqrt((2* log(self.counter))/n))) for x, n  in zip(self.averages, self.reps)]
      variances = [(numpy.var(x)) for x in self.rewards_of_plays]
      c = 1
      zeta = 1.2
      E = lambda t: zeta * log10(t)
      max_reward = 1.0
      self.ucb = [(x - (sqrt(2 * var * E(sum(self.reps)) / n ) \
                     - c * max_reward * E(sum(self.reps))))       \
                     for x, var, n  in zip(self.averages, variances, self.reps)]
      print self.ucb
      if self.ucb_init_done():
#          if self.eval_order == list(argsort(self.ucb)): # if nothing changed
#              if self.sendout_old == self.sendout :
#                  self.eval_order = list(argsort(self.ucb))[0:2] 
#              else:
#                  self.eval_order = list(argsort(self.ucb)) 
#          else:
              self.eval_order = list(argsort(self.ucb)) 
#          self.sendout_old = self.sendout
      else:
          self.eval_order.extend(list(argsort(self.ucb)))
          self.eval_order = self.eval_order[::self.random_popsize]
      
      print 'self.eval_order'
      print self.eval_order
      
          
          
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
       print 'pvector'
       print pvector
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
