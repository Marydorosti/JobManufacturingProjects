import numpy as np
import random
import pandas as pd
import time

from deap import base
import timeit
from deap import creator
from deap import tools
import ALBWithDevicetime
knapsack = ALBWithDevicetime.Knapsack01Problem()
# constants:
DIMENSIONS = len(knapsack)
POPULATION_SIZE = 500
MAX_GENERATIONS = 5
MIN_START_POSITION, MAX_START_POSITION = -5, 5
MIN_SPEED, MAX_SPEED = -3, 10
MAX_LOCAL_UPDATE_FACTOR = MAX_GLOBAL_UPDATE_FACTOR = 2.0

# set the random seed:
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

toolbox = base.Toolbox()


def create_number(a,b):
    import random
    w=random.sample(range(a), b)
    #random.shuffle(w)    
    #w=[random.randint(65,knapsack.NumberOfNodes-2)]
    for i in range(0,len(w)):
        if w[i] > knapsack.NumberOfNodes-3:
            w[i]=int(np.random.randint(knapsack.NumberOfNodes-3, size=1))
    return w
    #for i in w:
       #return i






# define a single objective, minimizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(0.7,0.3))

# define the particle class based on ndarray:
creator.create("Particle", np.ndarray, fitness=creator.FitnessMin, speed=None, best=None)

# create and initialize a new particle:
def createParticle():
    particle = creator.Particle(np.random.randint(MIN_START_POSITION,
                                                  MAX_START_POSITION,
                                                  DIMENSIONS))
    #particle=creator.Particle(create_number(len(knapsack),len(knapsack)))
    particle=creator.Particle(random.sample(range(len(knapsack)),len(knapsack)))                                              
    
    particle.speed = np.random.randint(MIN_SPEED, MAX_SPEED, DIMENSIONS)
    return particle

# create the 'particleCreator' operator to fill up a particle instance:
#toolbox.register("particleCreator", create_number,len(knapsack),len(knapsack))
toolbox.register("particleCreator", createParticle)


# create the 'population' operator to generate a list of particles:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.particleCreator)


def updateParticle(particle, best):

    # create random factors:
    localUpdateFactor = np.random.uniform(0, MAX_LOCAL_UPDATE_FACTOR, particle.size)
    globalUpdateFactor = np.random.uniform(0, MAX_GLOBAL_UPDATE_FACTOR, particle.size)

    # calculate local and global speed updates:
    localSpeedUpdate = localUpdateFactor * (particle.best - particle)
    globalSpeedUpdate = globalUpdateFactor * (best - particle)

    # scalculate updated speed:
    particle.speed = particle.speed + (localSpeedUpdate + globalSpeedUpdate)

    # enforce limits on the updated speed:
    particle.speed = np.clip(particle.speed, MIN_SPEED, MAX_SPEED)

    # replace particle position with old-position + speed:
    particle[:] = particle + particle.speed


toolbox.register("update", updateParticle)


# Himmelblau function:
#def himmelblau(particle):
    #x = particle[0]
    #y = particle[1]
    #f = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    #return f,  # return a tuple
def evaluate(particle):
    
    '''graph=knapsack.get_sum_of_column_violaton(individual[0])[4]
    g=GraphClass.Graph(graph)
    source = 0; sink = 6
    max_flow=g.FordFulkerson(source, sink)
    max_flow=0'''
    
    #f=[individual]
    #return knapsack.GET_COST(individual)[4],
    return knapsack.GET_COST(particle)[4],knapsack.GET_COST(particle)[1],

toolbox.register("evaluate", evaluate)


def main():
    # create the population of particle population:
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    best = None
    start_time = time.time()
    for generation in range(MAX_GENERATIONS):

        # evaluate all particles in polulation:
        for particle in population:

            # find the fitness of the particle:
            particle.fitness.values = toolbox.evaluate(particle)

            # particle best needs to be updated:
            if particle.best is None or particle.best.size == 0 or particle.best.fitness < particle.fitness:
                particle.best = creator.Particle(particle)
                particle.best.fitness.values = particle.fitness.values

            # global best needs to be updated:
            if best is None or best.size == 0 or best.fitness < particle.fitness:
                best = creator.Particle(particle)
                best.fitness.values = particle.fitness.values

        # update each particle's speed and position:
        for particle in population:
            toolbox.update(particle, best)

        # record the statistics for the current generation and print it:
        logbook.record(gen=generation, evals=len(population), **stats.compile(population))
        print(logbook.stream)


    from collections import defaultdict
    
    
    
    
    def list_duplicates(seq):
                tally = defaultdict(list)
                for i,item in enumerate(seq):
                     tally[item].append(i)
                return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>0)
    b=[]
    for d in sorted(list_duplicates(best)):
        b.append(d)
    #x=[]
    y=[]
    x=[]
    type=list(knapsack.type)
    #l=[]
    #for z in range(0,len(b)):
    for z,m in zip((range(0,len(b))),(range(0,len(type)))):    
        l=[]
        j=[]
       
        
        for c in b[z][1]:
            #for n in b[z][0]:
             #su=0
             l.append(knapsack.items[c][0])
             #s=knapsack.items[c][z+1]
             s=knapsack.items[c][int(type[z])]
             j.append(s)
           
        #if knapsack.time[z]>90:
                 #x.append((b[z][0],l,sum(j)))
               
        #else:
                 #x.append((b[z][0],l,max(j)))  
#####################################################################################        
        x.append((b[z][0],l,sum(j)))
        #y=[]
        #for i in range(0,len(x)):
                  #y.append(x[i][2])
        #x.append((b[z][0],l,sum(j)))
        #y.append((b[z][0],l,sum(j)))
        #y.append((z,l,sum(j)))
    
#print("--- %s seconds ---" % (time.time() - start_time))      

    df = pd.DataFrame(x, columns=['NodeNumber','OperatorsName','SumOfSpeed'])
    df.to_csv('output.csv')
    print(df) 
    # print info for best solution found:
    print("-- Best Particle = ", best)
    print("-- Best Fitness = ", best.fitness.values[0])
    print("DeviceCountViolaton:",knapsack.GET_COST(best)[0])
    print("--labors for each Node:",x,b)
    
    print("time is:",timeit.timeit(lambda: knapsack.GET_COST(best)[0], number=1))
    #print("Max_Flow:",knapsack.GET_COST(best)[1])
    print("SumOfRowColumnViolations:",knapsack.GET_COST(best)[2])
    
    print("time is:",timeit.timeit(lambda: knapsack.GET_COST(best)[2], number=1))
    
    
    
    
    print("Variety_violation:",knapsack.GET_COST(best)[3])
    print("time is:",timeit.timeit(lambda: knapsack.GET_COST(best)[3], number=1))
    
    
    print("violationS:",knapsack.GET_COST(best)[1])
    print("time is:",timeit.timeit(lambda: knapsack.GET_COST(best)[1], number=1))
    
    
    
    print("sumrOW-min(Min_table):",knapsack.get_sum_of_column_violaton(best)[1])
    print("time is:",timeit.timeit(lambda: knapsack.get_sum_of_column_violaton(best)[1], number=1))
    
    
    
    
    print("SumRowlistN:",knapsack.get_sum_of_column_violaton(best)[2])
    print("time is:",timeit.timeit(lambda: knapsack.get_sum_of_column_violaton(best)[2], number=1))
    
    
    print("sumCol:",knapsack.get_sum_of_column_violaton(best)[3])
    print("time is:",timeit.timeit(lambda: knapsack.get_sum_of_column_violaton(best)[3], number=1))
    
    
    #print("graph1:",knapsack.get_sum_of_column_violaton(best)[4])
    print("H:",knapsack.get_sum_of_column_violaton(best)[5])
    print("time is:",timeit.timeit(lambda: knapsack.get_sum_of_column_violaton(best)[5], number=1))
    
    
    print("Min_Table:",knapsack.get_sum_of_column_violaton(best)[6])
    print("time is:",timeit.timeit(lambda: knapsack.get_sum_of_column_violaton(best)[6], number=1))
    
    
    #print("S:",knapsack.get_sum_of_column_violaton(best)[8])
    print(len(knapsack.type))
    
    
    print("--- %s seconds ---" % (time.time() - start_time))
    


if __name__ == "__main__":
    main()