#%%time
import timeit
#start = timeit.timeit()
from deap import base
from deap import creator
from deap import tools

import random
import array

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ALBWithDevicetime
import cProfile
from joblib import Parallel ,delayed
start=timeit.timeit()
#import tsp
import elitism

# set the random seed for repeatable results
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# create the desired traveling salesman problem instace:
TSP_NAME = "bayg29"  # name of problem
#tsp = tsp.TravelingSalesmanProblem(TSP_NAME)
knapsack = ALBWithDevicetime.Knapsack01Problem()
# Genetic Algorithm constants:
POPULATION_SIZE = 2000
MAX_GENERATIONS =600
HALL_OF_FAME_SIZE = 50
P_CROSSOVER = 0  # probability for crossover
P_MUTATION = 0.35 # probability for mutating an individual

#************defination of function for createindividual with specific condition*************************
def create_number(a,b):
    import random
    w=random.sample(range(a), b)
    #random.shuffle(w)    
    
    #w=[random.randint(65,knapsack.NumberOfNodes-2)]
    for i in range(0,len(w)):
        if w[i] > knapsack.NumberOfNodes-3:
            w[i]=int(np.random.randint(knapsack.NumberOfNodes-3, size=1))
    return w

#*********************************************************************************************************





toolbox = base.Toolbox()

# define a single objective, minimizing fitness strategy:
#creator.create("FitnessMin", base.Fitness, weights=(0.65,0.35,))
#creator.create("FitnessMin", base.Fitness, weights=(0.65,0.35))
creator.create("FitnessMin", base.Fitness, weights=(1,))

# create the Individual class based on list of integers:
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

# create an operator that generates randomly shuffled indices:
#toolbox.register("randomOrder", random.sample, range(len(knapsack)), len(knapsack))
#toolbox.register("randomOrder", random.sample, range(len(knapsack)),len(knapsack))


toolbox.register("randomOrder", create_number,len(knapsack),len(knapsack))


#toolbox.register("randomOrder", random.randint, 0,50)
#toolbox.register("randomOrder", random.randint,1,knapsack.NumberOfNodes-2)
#toolbox.register("randomOrder", random.randint,1,knapsack.NumberOfNodes-2)

# create the individual creation operator to fill up an Individual instance with shuffled indices:
toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomOrder)

# create the population creation operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# fitness calculation - compute the total distance of the list of cities represented by indices:
#def tpsDistance(individual):
    #return tsp.getTotalDistance(individual),  # return a tuple

def evaluate(individual):
    
    '''graph=knapsack.get_sum_of_column_violaton(individual[0])[4]
    g=GraphClass.Graph(graph)
    source = 0; sink = 6
    max_flow=g.FordFulkerson(source, sink)
    max_flow=0'''
    
    #f=[individual]
    return knapsack.GET_COST(individual)[1],
    #return knapsack.GET_COST(individual)[4],knapsack.GET_COST(individual)[1],
toolbox.register("evaluate", evaluate)


# Genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/len(knapsack))
#toolbox.register("mutate", tools.mutUniformInt,10,20, indpb=1.0/len(knapsack))


# Genetic Algorithm flow:
def main():

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max",np.max)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # print best individual info:
    #best = hof.items[0]
    best = tools.selBest(population, 1)[0]
    
    from collections import defaultdict
    
    
    
    
    #def list_duplicates(seq):
                #tally = defaultdict(list)
                #for i,item in enumerate(seq):
                     #tally[item].append(i)
               # return ((key,locs) for key,locs in tally.items() 
                            #if len(locs)>0)
    #b=[]
    #for d in sorted(list_duplicates(best)):
        #b.append(d)
    #x=[]
    #y=[]
    #x=[]
    #type=list(knapsack.type)
    #l=[]
    #for z in range(0,len(b)):
    #for z,m in zip((range(0,len(b))),(range(0,len(type)))):    
        #l=[]
        #j=[]
       
        
        #for c in b[z][1]:
            #for n in b[z][0]:
             #su=0
             #l.append(knapsack.items[c][0])
             #s=knapsack.items[c][z+1]
             #s=knapsack.items[c][int(type[z])]
             #j.append(s)
           
        #if knapsack.time[z]>90:
                 #x.append((b[z][0],l,sum(j)))
               
        #else:
                 #x.append((b[z][0],l,max(j)))  
#####################################################################################        
        #x.append((b[z][0],l,sum(j)))
        #y=[]
        #for i in range(0,len(x)):
                  #y.append(x[i][2])
        #x.append((b[z][0],l,sum(j)))
        #y.append((b[z][0],l,sum(j)))
        #y.append((z,l,sum(j)))
        
        
        
    #t = timeit.timeit(lambda: print_square(3), number=10)
  
# printing the execution time
    #print(t)

#***********calculate finish time of project***************************************    
        
    def calculate_finish_time(best):
        
        #N=knapsack.GET_COST(best)[1]
        #time= [[0 for _ in range(knapsack.NumberOfNodes)] for _ in range(knapsack.NumberOfNodes)]
        graph1=knapsack.get_sum_of_column_violaton(best)[4]
        rows = len(graph1)  
        cols = len(graph1[0])
        for i in range(1, rows-1):
             for j in range(0, cols):
        #for i , j in zip(range(1, rows-1),range(0, cols)):         
                 if graph1[i][j]!=0:
                    graph1[i][j]=knapsack.InputTime*knapsack.NumberOfgoods/graph1[i][j]
                    #graph1[i][j] = "{:.2f}".format()
                    #graph1[i][j]=int(graph1[i][j])
                    graph1[i][j]=round(graph1[i][j], 2)
        '''def creategraph1(i,j) :
             if graph1[i][j]!=0:
                    graph1[i][j]=knapsack.InputTime*knapsack.NumberOfgoods/graph1[i][j]
                    #graph1[i][j] = "{:.2f}".format()
                    #graph1[i][j]=int(graph1[i][j])
                    graph1[i][j]=round(graph1[i][j], 2)
             return (graph1)
        graph1=Parallel(n_jobs=2)(delayed(creategraph1)(i,j)for i in range(1, rows-1) for j in range(0, cols)  )'''     
        #FINISHtime=[0 for _ in range(knapsack.NumberOfNodes)]
        
        #p=[k for k, e in enumerate(graph1[0]) if e != 0] 
        #for i in range(0,  rows):
           #FINISHtime[]  
        #for i in range(0,len(FINISHtime)):    
               #for P in p:
        time = np.copy(graph1)           #for
        S=[]
        N=[]           
        for i in range(1, rows-1):
             ColumnList=[]
             #sumRow=0
             RowList=[]
             sumCol=0
             sumRow=0
             for j in range(1, cols):
                 RowList.append(time[i][j])
                 ColumnList.append(time[j][i])
                 #sumCol=sumCol+graph1[j][i]
                 sumCol=sumCol+time[j][i]
                 sumRow=sumRow+time[i][j]
             S.append(ColumnList)
             N.append(sumRow) 
        #a=[]
        
        H=knapsack.get_sum_of_column_violaton(best)[5]
        
        q=[0 for _ in range(len(N))]
        
        #for i in range(0,len(N)) :
            
            #q[i]=N[i]+max(S[i])
        for i in range(0,len(H)):
            if H[i][0]==0:
                    
                    q[i]=N[i]
            else:
            #q[i]=N[i]+max(graph1[b][i+1] for b in H[i] )
             q[i]=N[i]+max(q[b-1] for b in H[i] )
             
             q[i]=round(q[i], 2)
            
            #a.append(q)
        return  q,S,N,len(S),(max(q)+knapsack.InputTime)
            
            
            
        #for i in range(0,len)
        
        
        
        
        
        
        
        
        
    #x=knapsack.get_sum_of_column_violaton(best)[9] 
    #start_time = time.time()   
        
    #import pandas as pd
    #import timeit
        
    #df = pd.DataFrame(x, columns=['NodeNumber','OperatorsName','SumOfSpeed'])
    #df.to_csv('output.csv')
    #print(df)   
    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])
   
    #print("-- Best Ever Fitness = ", hof.items[1])

    print("-- max_flow/violation = ",best.fitness.values[0])
    #print("labors:","B:"+str(knapsack.items[best[0]][0]),"C:"+str(knapsack.items[best[1]][0]),"D:"+str(knapsack.items[best[2]][0])+" "+str(knapsack.items[best[3]][0]),"E:"+str(knapsack.items[best[4]][0])+" "+str(knapsack.items[best[5]][0]))
   # knapsack.printItems(best)
    #print(knapsack.gett_Value2(individual)[1])
    # extract statistics:
    #print("--- %s seconds ---" % (time.time() - start_time))    
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")
    #print("--labors for each Node:",x,b)
    
    print("DeviceCountViolaton:",knapsack.GET_COST(best)[0])
    
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
    
    
    #print("Min_Table:",knapsack.get_sum_of_column_violaton(best)[6])
    print("time is:",timeit.timeit(lambda: knapsack.get_sum_of_column_violaton(best)[6], number=1))
    #print("time is:",timeit.timeit(lambda: knapsack.get_sum_of_column_violaton(best)[9], number=1))
    print("outputx:",knapsack.get_sum_of_column_violaton(best)[9])
    print("UTPUT:",knapsack.print_output(best))
    knapsack.print_output(best).to_csv('output.csv')
    print("FinishTimeIs:",calculate_finish_time(best)[0],calculate_finish_time(best)[1],calculate_finish_time(best)[2],calculate_finish_time(best)[3],calculate_finish_time(best)[4])
    #print("outputy:",knapsack.get_sum_of_column_violaton(best)[10])
    #print("OUTPUT:",knapsack.GET_COST(best)[9])
    #print("first level unemployment time:",4*knapsack.W[np.argmin(knapsack.get_sum_of_column_violaton(best)[11])]//min(knapsack.get_sum_of_column_violaton(best)[11]))
    
    
    #print("S:",knapsack.get_sum_of_column_violaton(best)[8])
    print(len(knapsack.type))
    
    
    
    
    
    
    
    
    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])

    # plot best solution:
    #plt.figure(1)
    #tsp.plotData(best)

    # plot statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")
    plt.figure(2)
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')

    # show both plots:
    plt.show()


if __name__ == "__main__":
   # main()
   #cProfile.run('main()')
    #import cProfile, pstats
    #profiler = cProfile.Profile()
    #profiler.enable()
    
    main()
    end = timeit.timeit()
    print(end - start)
    #profiler.disable()
    #stats = pstats.Stats(profiler).sort_stats('ncalls')
    #stats.print_stats()
   
   
   
   
   
   
   
   
