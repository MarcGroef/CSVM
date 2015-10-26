import random
from numpy import array, log, ones, random, log10, argmin
from param_generator import ParameterGenerator
import xnes

generators = ['xNES_bare']

class xNES_bare(ParameterGenerator):
    """
    The CMAES_advanced subclass of ParameterGenerator
    returns the next 
    """
    def __init__(self):
        """The constructor initializes empty lists"""
        super(xNES_bare, self).__init__()
        self.xnes = None
        self.fitnesses = []
        self.solutions = []
        self.available = []
        self.minimum = []
        self.maximum = []
        self.best = 1000
        self.best_params = None
        


    def psetup_cmaes(self):
        xstart = []
        
        for idx in range(len(self.param_names)):
            if self.param_settings[idx]['type'] == "static":
                continue
            self.minimum.append(self.param_settings[idx]['min'])
            self.maximum.append(self.param_settings[idx]['max'])
            settings = self.param_settings[idx]
            name = self.param_names[idx]
            if settings['scaling'] == "linear":
                print "Linear scaling for parameter %s. \n Shown are the bounds between which values have the same sensitivity " %name
                bounds =[(self.reverse_linear_scaling(self.minimum[-1],
                                                      self.maximum[-1],
                                                      x )) for x in range(11)]
                print [("%.2f" %x) for x in bounds]
            elif settings['scaling'] == "log":
                print "Log scaling for parameter %s. \n Shown are the bounds between which values have the same sensitivity " %name
                bounds =  [(self.reverse_log_scaling(self.minimum[-1],
                                                     self.maximum[-1],
                                                     x )) for x in range(11)]
                print [("%.2e" %x) for x in bounds]

              



        print "Number of dimensions: %d" % len(self.minimum)
        
       
        xstart =50*ones(len(self.minimum))  # start in the middle of searchspace 
        xstart =ones(len(self.minimum))  # start in the middle of searchspace 
        self.xnes = xnes.xNES(xstart) 

        

    def set_result(self,params, result, id=0):
        """
        This method sets the result/evalution of the last set of
        parameters. This can be used by subclasses to generate the next
        value for each parameter.
        """
        self.fitnesses[id] = result
        corrected_result = None
        if self.generation_done():
            negated_fitnesses = [(-1*x) for x in self.fitnesses]
            print "best " + repr(self.fitnesses)
            #self.xnes.tell(self.solutions, self.fitnesses)
            self.xnes.tell(self.solutions, negated_fitnesses)
            gen_best = min(self.fitnesses)
            gen_best_idx = argmin(self.fitnesses)
            gen_best_params = self.solutions[gen_best_idx]
            corrected_result = (gen_best_params, gen_best)
            print 'corrected_result' + repr(corrected_result)

            if min(self.fitnesses) < self.best:
                self.best = min(self.fitnesses)
                self.best_params = self.solutions[argmin(self.fitnesses)]
                #self.to_file()
                print "new best found: " +repr(self.best) + " " + repr(self.to_outside_repr(self.best_params)) 
        print 'corrected_result' + repr(corrected_result)
        return corrected_result

    def get_best_result(self):
         if self.best is None:
             best = min(self.fitnesses)
             best_idx = argmin(self.fitnesses)
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
        self.solutions = self.xnes.ask()
        self.solutions = [[(self.clamp(elem, 0, 10)) for elem in solution] for solution in self.solutions] 
        
        self.fitnesses = []
        self.param_values = []
        self.available = range(len(self.solutions))
        for i in enumerate(self.solutions):
            self.fitnesses.append(None)
            self.param_values.append([])

    def generation_done(self):
        """Check if the current generation finished evaluating"""
        done = True
        remain = []
        for idx, f in enumerate(self.fitnesses):
            if f == None:
                done = False
                remain.append(idx)
        if remain:
            print "Still waiting for %s" % repr(remain)
        return done

    def generate_parameters(self, id=0):
        """
        This method gives a new value for each parameter using
        CMA. If no solutions are available for evaluating yet,
        a new set is retrieved, otherwise the cached values are
        used.
        """
        if not self.xnes:
            self.psetup_cmaes()

        if self.generation_done():
            self.get_new_set()

        if not self.available:
            return 0, None   # Nothing to process yet

        k = self.available.pop(0)
        print "Found set %d out of %d" % (k, len(self.solutions))

        pvector = self.solutions[k]
        vector_counter = 0
        for idx in range(len(self.param_names)):
            name = self.param_names[idx]
            settings = self.param_settings[idx]
            if settings['type'] == "static":
                new_value = settings['value'];
                print "Parameter %s: %d" % (name, new_value)
            #elif name == 'D2' or name == 'COST' or name == 'nHiddenV' or name == 'SVM_C' or name == 'ALPHA_ITER':
            elif settings['scaling'] == "linear":

                new_value = self.reverse_linear_scaling(self.minimum[vector_counter],
                                                        self.maximum[vector_counter],
                                                        pvector[vector_counter])
                vector_counter += 1
                print "Parameter %s: %.5f" % (name, new_value)
            elif settings['scaling'] == "log":
                new_value = self.reverse_log_scaling(self.minimum[vector_counter],
                                                     self.maximum[vector_counter],
                                                     pvector[vector_counter])
                vector_counter += 1

            self.param_values[k].append(new_value)
        
        return k, self.param_values[k]

    def get_next_parameter(self, name, value, settings, id=0):
        pass

   
    
    def reverse_linear_scaling(self,minimum,maximum,y):
       """
       parameters from linear scaling back to normal so they can be handed over to be evaluated
       """
       value = minimum + y * (maximum-minimum)/10.0
       return value
    
    def reverse_log_scaling(self,minimum,maximum,y):
       """
       parameters from logspace back to normal so they can be handed over to be evaluated
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

    def to_file(self):
        params = self.to_outside_repr(self.best_params)
        params = [(x[1]) for x in params]
        #f = open("../NumPer/Q_%f"%self.best, 'w')
        f = open("NumPer/Q_%f"%self.best, 'w')
        f.write("world\n")
        f.write("discreteActions\t%d\n" % params[0])
        f.write("stateDimension\t%d\n" % params[1])
        f.write("actionDimension\t%d\n" % params[2])
        f.write("\nalgorithm\n")
        f.write("alpha\t\t\t%.15f\n" % params[3])
        f.write("beta\t\t\t%.15f\n" % params[4])
        f.write("COST\t\t\t%.15f\n" % params[5])
        f.write("D2\t\t\t\t%.15f\n" % params[6])
        f.write("SVM_C\t\t\t%d\n" % params[7])
        f.write("ALPHA_ITER\t\t%d\n" % params[8])
        f.write("NR_REP1\t\t\t%d\n" % params[9])
        f.write("NR_REP2\t\t\t%d\n" % params[10])
        f.write("EPS\t\t\t\t%.15f\n" % params[11])
        f.write("\nnn\n")
        f.write("nHiddenV\t\t%d\n" % params[12])
        f.close()

