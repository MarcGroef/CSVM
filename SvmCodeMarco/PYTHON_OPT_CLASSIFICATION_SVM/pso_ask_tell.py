import sys
import scipy
import numpy
import itertools as i
from numpy import array
import math 
MAX_VAL = 10e10 

#import matplotlib.font_manager
#leg_prop = matplotlib.font_manager.FontProperties(size=8)

#import matplotlib
#matplotlib.use('Agg')

def generate_particles(pop_size, lower, upper): # args takes lower and upper bounds for parameters
  particles = array([[scipy.random.uniform(lower[i], upper[i]) for i in range(len(lower))] for _ in range(pop_size)], dtype=numpy.longdouble)
  #particles = [((eval_fitness(x),array(x))) for x in particles]
  particles = [(None,array(x)) for x in particles]
  return particles

def initialize_velocities(pop_size, vmin, vmax):
  velocities = [array([(scipy.random.uniform(vmin[i], vmax[i])) for i in range(len(vmin))],dtype=numpy.longdouble) for _ in range(pop_size)]
  return velocities

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(i.islice(iterable, n))


class pso():


  def __init__(self, pop_size, lower_bound, upper_bound,ac1, ac2, w, max_evals):
    self.pop_size = pop_size
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound
    self.ac1 = ac1
    self.ac2 = ac2
    self.w = w
    self.max_evals = max_evals
    self.dimension = len(lower_bound)
    self.population = generate_particles(self.pop_size, self.lower_bound, self.upper_bound)
    self.initial_population = self.population[:]
    self.individual_best = [(None) for _ in range(self.pop_size)] 
    self.d = numpy.fabs(self.lower_bound - self.upper_bound)
    self.vmax = self.d
    self.vmin = -self.d
    self.velocities = initialize_velocities(self.pop_size, self.vmin, self.vmax)
    self.evalcounter = 0
    #self.global_best = min(self.population)
    #self.global_best = self.population[0]  # just for now
    self.global_best = None
    self.inial_population_evaluated = False
    self.eval_index = None
    self.counter = 0
    self.idx = -1
    self.evolution_of_best = []
    
  def ask(self, number): # evaluate number offspring at once
    print 'ask'
    a =   ((self.move()) for _ in i.count())
    #b =  ((elem) for elem in a if elem != None)
    b =  i.takewhile(lambda x: x!=None, a)
    #if number <= len(b):
    solutions = take(number, b)

    print "solutions " + repr(solutions) 
    return solutions
    
    #return [(self.move()) for _ in range(number)]

  def tell(self, solutions, values):
    print 'tell'
    print 'values ' + repr(values)
    for solution, fitness in zip(solutions, values):
      index = solution[0]
      position = solution[1]
      self.incorporate_fitval(index, position, fitness)

  def move(self):
    #if self.evalcounter == self.max_evals: # how tell that maximum number of evals is reached?
      if not self.inial_population_evaluated:
          if len(self.initial_population) > 0:
              print "evaluating initial population"
              fitness, particle = self.initial_population.pop(0)
              self.idx += 1
              self.individual_best[self.idx] = (fitness, particle) # in initial run no better position than the initial position is known
              return (self.idx, particle)
          else:
              print "initial eval done"
              print self.population
              print 'self.population'
              fitnesses = [(elem[0]) for elem in self.population]
              print 'fitnesses'
              print fitnesses
              if None in fitnesses:
                  return
              self.inial_population_evaluated = True
              print 'fitnesses ' + repr(fitnesses)
              self.global_best = self.population[fitnesses.index(min(fitnesses))]

              
#          for idx, element in enumerate(self.population):
#              fitness, particle = element
#              if fitness == None:
#                  self.eval_index = idx
#                  #self.population[idx] = (1000, particle) # so particle is not evaluated twice in initial run
#                  #self.individual_best[idx] = (1000, particle)
#                                                          # more elegant way?
#                  #print "Returning unevaluated particle %d" % idx
#                  print "evaluating initial population"
#                  return (idx,particle)
#          self.inial_population_evaluated = True
#          fitnesses = [(elem[0]) for elem in self.population]
#          self.global_best = self.population[fitnesses.index(min(fitnesses))]
          
       
      print 'initializing done'
      #self.index = scipy.random.randint(self.pop_size) # pick particles to be updated randomly
      self.index = self.counter % self.pop_size # pick particles to be updated randomly
      self.counter +=1
      phi_1 = scipy.random.uniform(0,self.ac1) 
      phi_2 = scipy.random.uniform(0,self.ac2) 
      particle = self.population[self.index][1]
      best_ind_particle = self.individual_best[self.index][1]
      print "best_ind_particle"
      print best_ind_particle

      print 'self.global_best ' + repr(self.global_best)
      #self.velocities[self.index] = self.w *self.velocities[self.index] + phi_2 * (self.global_best[1] - particle)# here, phi_x is a vector, in some implementations it is only a scalar
      self.velocities[self.index] = self.w *self.velocities[self.index] + phi_1 * (best_ind_particle - particle) + phi_2 * (self.global_best[1] - particle)
      self.bound(self.velocities[self.index], self.vmin, self.vmax)
      self.new_particle_position = particle + self.velocities[self.index]
      self.bound(self.new_particle_position, self.lower_bound, self.upper_bound)
      return (self.index, self.new_particle_position)
      
  def incorporate_fitval(self, index, solution, fitness): 
      if not self.inial_population_evaluated:
      #    old_fitness, particle = self.population[self.eval_index]
          self.population[index] = (fitness, solution)
          self.individual_best[index] = self.population[index]
      #    print "Storing unevaluated particle %d" % self.eval_index
          return
      self.population[index] = (fitness, solution)
      if self.individual_best[index]== None:
          self.individual_best[index] = self.population[index]
          
      

      if fitness < self.individual_best[index][0]:
          self.individual_best[index] = self.population[index]

      if self.population[index][0] < self.global_best[0]:
          self.global_best = self.population[index]
          self.evolution_of_best.append((self.evalcounter, self.global_best))
          print 'X' + repr(self.evalcounter) + ' ' + repr(self.global_best)

      self.evalcounter +=1 

     

  def bound(self, orig, lower, upper):
      for elem in range(len(lower)):
          if orig[elem] < lower[elem]:
              orig[elem] = lower[elem]
          if orig[elem] > upper[elem]:
              orig[elem] = upper[elem]

if __name__ == '__main__':
    runs = 50
    evaluations = 4000
    swarm_size = 183
    phi_global = 3.0539
    w = -0.2797
    bests = numpy.zeros(evaluations + 1)
    p = pso1('dummy', swarm_size,array([-10, -10]),array([10,10]), phi_global, w, evaluations)
    for _ in range(runs): # run using parameter from the "good params" paper
      p.move()
      print p.new_particle_position
      p.incorporate_fitval(1)



    

 



