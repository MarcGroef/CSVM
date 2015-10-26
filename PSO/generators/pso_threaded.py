import random
from numpy import array, log, ones, random, log10, argmin
from param_generator import ParameterGenerator
import pso_ask_tell

generators = ['PSO_without_bandit']

class PSO(ParameterGenerator):
    """
    The PSO subclass of ParameterGenerator
    returns the next 
    """
    def __init__(self):
        """The constructor initializes empty lists"""
        super(PSO, self).__init__()
        self.pso = None
        self.fitnesses = []
        self.solutions = []
        self.indexes = None
        self.positions = None 
        self.available = []
        self.minimum = []
        self.maximum = []
        self.first_gen = True
        self.best = 1000
        self.best_params = None
        self.evals_at_once = None

    def setup_pso(self):
        evaluations = 2000
        swarm_size = 64
        #swarm_size = 10 # for quicker debugging, don't forget to remove
        phi_global = 0.6239
        phi_individual = 1.6319
        w = 0.6571
        self.evals_at_once = 5
        for idx in range(len(self.param_names)):
            if self.param_settings[idx]['type'] == "static":
                continue
            self.minimum.append(self.param_settings[idx]['min'])
            self.maximum.append(self.param_settings[idx]['max'])

        print "Number of dimensions: %d" % len(self.minimum)
        dim = len(self.minimum)

        #self.pso = pso_ask_tell.pso(swarm_size, array(minimum), array(maximum), phi_global, w, evaluations)
                # (version wihout scaling)
        self.pso = pso_ask_tell.pso(swarm_size, array(dim*[0]), array(dim*[10]),phi_individual, phi_global, w, evaluations)
        self.param_values = []
        self.result = []
 


    def set_result(self,params, result, id=0):
         """
         This method sets the result/evalution of the last set of
         parameters. This can be used by subclasses to generate the next
         value for each parameter.
         """
         self.fitnesses[id] = result
         corrected_result = None
         if self.generation_done():
             self.pso.tell(self.solutions, self.fitnesses)
             gen_best = min(self.fitnesses)
             gen_best_idx = argmin(self.fitnesses)
             gen_best_params = self.solutions[gen_best_idx]
             corrected_result = (gen_best_params, gen_best)
             print 'corrected_result' + repr(corrected_result)
 
             if min(self.fitnesses) < self.best:
                 self.best = min(self.fitnesses)
                 self.best_params = self.solutions[argmin(self.fitnesses)]
                 #self.to_file()
                 print "new best found: " +repr(self.best) + " " + repr(self.to_outside_repr(self.best_params[1]))
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
         vector = [(x[1]) for x in self.to_outside_repr(best_params[1])]
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
        self.solutions = self.pso.ask(self.evals_at_once) # returns  tuples of the index  and position of particle 
        self.indexes = [(solution[0]) for solution in self.solutions]
        print 'self.indexes ' + repr(self.indexes)
        self.positions = [(solution[1]) for solution in self.solutions]
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
        if not self.pso:
            self.setup_pso()

        if self.generation_done():
            self.get_new_set()

        if not self.available:
            return 0, None   # Nothing to process yet

        k = self.available.pop(0)
        print "Found set %d out of %d" % (k, len(self.solutions))
        pvector = self.positions[k]
        new_value = [(x[1]) for x in self.to_outside_repr(pvector)]
        self.param_values[k].extend(new_value)
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


 
    def to_file(self):
        params = self.to_outside_repr(self.best_params)
        params = [(x[1]) for x in params]
        #f = open("../NumPer/Q_%f"%self.best, 'w') # change to this line when using command line version
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
         
    def to_outside_repr(self, pvector):
        """
        converts scaled output back to the actual parameter space
        """
        #pvector = pvector[1]
        vector_counter = 0
        out = []
        for idx  in range(len(self.param_names)):
            print vector_counter
            print 'pvector'
            print pvector
            print 'pvector[]'
            print pvector[vector_counter]
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




