# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 09:34:39 2023

@author: m.karami
"""

from deap import base
from deap import creator
from deap import tools
import random
import timeit
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import elitism
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
import traceback
import sys
from flask import Flask, request, render_template, session, redirect
from flask import Flask, redirect, url_for, request
import json
import pyodbc
import pandas as pd
from datetime import timedelta
import numpy as np
import random
import array
import statistics
import time
from statistics import mean
from collections import defaultdict
import pandas as pd
import random
from joblib import Parallel ,delayed

app = Flask(__name__,template_folder='Template')

class Knapsack01Problem:
    """This class encapsulates the Knapsack 0-1 Problem from RosettaCode.org
    """

    def __init__(self,af,df,Device,settings):

        # initialize instance variables:
        self.items = []
        self.maxCapacity = 0
        self.af=af
        self.df=df
        self.device=Device
        self.settings=settings

        # initialize the data:
        self.__initData()

    def __len__(self):
        """
        :return: the total number of items defined in the problem
        """
        return len(self.items)
        

    def __initData(self):
        """initializes the RosettaCode.org knapsack 0-1 problem data
        
        """
        
        
        
        #self.af=pd.read_excel('a.xlsx','OperatorsInformation')
        
        #self.af=pd.read_excel('a.xlsx','OperatorsInformation')
        #****operators information******
        #self.af=pd.DataFrame(self.af)
        #self.af=self.af[0:70]
        
        #self.af=list(self.af)
        #self.records = self.af.to_records(index=False)
        #self.af=pd.DataFrame.from_dict(self.af, orient='index')
        #self.af=self.af.transpose()
        self.af = pd.DataFrame({ key:pd.Series(value) for key, value in self.af.items() })
        self.records = self.af.to_records(index=False)
        self.items = list(self.records)
        #self.items = list(self.af)
        
        self.InputTime=60
        self.NumberOfgoods=400
        
        
        
        
        
        
        
        
        #self.af=pd.read_excel('input.xlsx','OperatorsInformation')
        #self.af=pd.read_excel('a.xlsx','OperatorsInformation')
        
        #self.af=pd.read_excel('a.xlsx','OperatorsInformation')
        #self.af=self.af[0:70]
        #self.records = self.af.to_records(index=False)
        #self.items = list(self.records)
        
        self.InputTime=60
        self.NumberOfgoods=400
        
        self.df = pd.DataFrame({ key:pd.Series(value) for key, value in self.df.items() })
        #string_filter = [s for s in self.df.iloc[:,3] if s is not float]
        string_filter=self.df[self.df['NodesJobType'].isnull()]
        int_filter=self.df[self.df['NodesJobType'].notnull()]
        #string_filter = self.df[:,3][self.df.iloc[:,3].isnull()]
        #int_filter = self.df[:,3][self.df.iloc[:,3].notnull()]
        string_filter1=list(string_filter.iloc[:,3])
        int_filter1=list(int_filter.iloc[:,3])
        #
        #int_filter = [s for s in self.df.iloc[:,3] if s is not 'nan']
        sortedlist=string_filter1+int_filter1
        #sortedlist=self.df.iloc[:,3][::-1]
        self.df['NodesJobType']=sortedlist
        
        k=self.df.loc[self.df['nodei']!= 0]
        
        
        self.m=list(k.iloc[:,0])
        
        self.n=list(k.iloc[:,1])
        
        #self.e=self.df.loc[self.df['nodei'] == 0]
        self.e=self.df
        self.f=list(self.e.iloc[:,0])
        self.g=list(self.e.iloc[:,1])
        
        
        self.device = pd.DataFrame({ key:pd.Series(value) for key, value in self.device.items() })
        #self.Device=pd.read_excel('input.xlsx','DeviceSum')
        #self.Device=pd.read_excel('a.xlsx','DeviceSum')
        self.device=list(self.device.iloc[:,1])
       
        
        
       
        
        #self.type=k.iloc[:,3]
        self.type=k['NodesJobType']
        
        
        
        
        
        #self.time=[16,33,32,19,43,31,0,45,26,70,42,107,
                   #63,117,54,23,63,33,45,60,36,90,110,81,45,45,40,33,60,90,45,48,73,111,111,0,58,86,51
                   #,130,85,122,80,99,54,193,80,125,100,100,93,80,60,48,54,115,54,93,39,1,1,56,1,1,1] 
        
        
        
        self.hardConstraintPenalty=10
        
        self.NumberOfNodes=len(self.df.nodei.unique())+1
    
        self.z= [[0 for _ in range(self.NumberOfNodes)] for _ in range(self.NumberOfNodes)]
        
        #self.settings=pd.read_excel('input.xlsx','AccuracySettings')
        self.settings = pd.DataFrame({ key:pd.Series(value) for key, value in self.settings.items() })
        
        self.accuracy=self.settings.iloc[0,0]
        
        self.sie=self.settings.iloc[0,1]
        self.sae=self.settings.iloc[0,2]
        
        W=self.df.loc[self.df['nodei'] == 0]
        self.W=list(W)
        
        
###################################################################################################       
                 
################****************#########################*************************#######################################33
    def Normal_attr_fltList(self,individual):
         
         for i in range(0,len(individual)):
            if individual[i] > self.NumberOfNodes-3:
            #if individual[i] > 70:
              individual[i]=int(np.random.randint(self.NumberOfNodes-3, size=1))
              #individual[i]=int(np.random.randint(70, size=1))
        
         return individual
        
        
####################***DEVICEVIOLATIONS**********************####################################


     
    def get_device_violations(self,attr_fltList):

       
         violations=0

           
         from collections import defaultdict
         def list_duplicates(seq):
                tally = defaultdict(list)
                for i,item in enumerate(seq):
                     tally[item].append(i)
                return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>0)                   
         hh=[]
         F=[]
         for d in sorted(list_duplicates(attr_fltList)):
                    hh.append(d)
                    
         for f in sorted(list_duplicates(self.type)):
                    F.append(f) 
                    
                    
                    

            
         k=[]                   
         for q in range(0,len(F)):
             
             sum=0
             
             
             for s in F[q][1]:
               if s <=len(hh)-1:
                 
                 sum+=len(hh[s][1])
                 #sum+=s
               k.append(sum)
                    
         for p,o in zip(k,self.device):
             if p>o:
                  violations=violations+(p-o)

         return violations           
         
         
################***************get sum of column violaton########################################
       
    def get_sum_of_column_violaton(self,attr_fltList):
        
       

           z=self.z
      
           
           violations=0
           #z=self.z
           
           for r , t ,b in zip(self.f,self.g,(self.e.iloc[:,2])):
               
                z[r][t]=b
           
           
           
           
           def list_duplicates(seq):
                tally = defaultdict(list)
                for i,item in enumerate(seq):
                     tally[item].append(i)
                return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>0)
           h=[]
           for dup in sorted(list_duplicates(attr_fltList)):
               h.append(dup)
           
           
           x=[]

           y=[]
           Type=list(self.type)
           for w,m in zip((range(0,len(h))),(range(0,len(Type)))):

             j=[]
             g=[] 
        
             for c in h[w][1]:
            

                 
                 s=self.items[c][int(Type[w])]
######################################################################################                
                 
                 j.append(s)
             
#####################################################################################        

             x.append(sum(j))

##########################################################################################                  
             

           for  s, p,v in zip(self.m,self.n,range(0,len(x))):

                     z[s][p]=x[v]

             
#############################################################################################             
          
             
           graph1=z
           


           rows = len(graph1)  
           cols = len(graph1[0]) 
           S=[]
           N=[]
          
           for i in range(1, rows-1):  
             #SumRow = []
             ColumnList=[]
             #sumRow=0
             RowList=[]
             sumCol=0
             sumRow=0
             #SumRow=0
             #N=[]
             #S=[]
             #F=[]
             
             
             for j in range(0, cols):
                 

                 RowList.append(graph1[i][j])
                 ColumnList.append(graph1[j][i])

                 sumCol=sumCol+graph1[j][i]
                 sumRow=sumRow+graph1[i][j]

                 
             S.append(ColumnList)
             N.append(sumRow)      
             
              
             
           p=[k for k, e in enumerate(graph1[0]) if e != 0] 
           
          
                   
           H=[]
           for r in range(0,len(S)):
               h=[k for k, e in enumerate(S[r]) if e != 0] 
               H.append(h)
           
           
           

                  
                  
          
           F = [0 for _ in range(len(N))]
           k=[0 for _ in range(len(N))]

              
              
           C=[]

           for i in range(0,len(N)):
                C.append(N[i]+5)    
              
                   
           B=[]        
           for  i  in range(0,len(N)):
               #for P in p:
               
                  #if i!= P-1:
                   
                      J=N[i]-min(graph1[b][i+1] for b in H[i] )
                      K=(C[i]-N[i])+(max(C[i]+graph1[b][i+1] for b in H[i] ))
                      
                      
                      
                      #k[i]=abs(K)
                      
                      
                      
                      F[i]=abs(J)
                      B.append(abs(J))
                      #if i!= P-1
                      #if abs(J)>5:
                      if abs(J)>self.accuracy:
                          #violations=violations+abs(J)
                          #violations=violations+1
                          violations=violations+(abs(j)-self.accuracy)
                          
                      else:
                          violations=violations+0
                          

           T=[]
           #for P in p:
           for i in range(0,len(N)):    
               for P in p:
                   if i== P-1:
                       T.append(F[i])

                       
           c=sum(F)
           g=sum(T)
           df=1

           if c>self.sae:
               violations=violations+1
               violations=violations+(c-self.sae)

           if g>self.sie:

               violations=violations+(g-self.sae)


            
           return violations,F,N,sumCol,graph1,H,g,S,x,y,c,df
           
                   
                   

             
             
           
#########################################################################################



    def get_Variety_violation(self,attr_fltList):
        

        violations=0
        a_set = set(attr_fltList)

        if len(a_set)==len(self.type):

            violations=0
        else:
            violations=violations+1
        return violations    
    
        
        
            
                 
############################*******GET_COST******#############################################3##         
        
    def GET_COST(self,attr_fltList):
        
             #attr_fltList=self.Normal_attr_fltList(individual)
             
             get_Variety_violation=self.get_Variety_violation(attr_fltList)
             
             #get_Variety_violation=self.get_Variety_violation(attr_fltList)
             
             #SumOfRowColumnViolations = self.gett_Value2(attr_fltList)[2]
             
             deviceViolations = self.get_device_violations(attr_fltList)
             
             #CountShiftViolationForEachGood=self.CountShiftViolationForEachGood(nurseShiftsDict)
             
             #Max_Flow = self.get_sum_of_column_violaton(attr_fltList)[6]
             
             sum_of_column_violaton=self.get_sum_of_column_violaton(attr_fltList)[0]
             
             hardContstraintViolations=sum_of_column_violaton+deviceViolations
             
             #hardContstraintViolations=get_Variety_violation
             
             #violations=self.hardConstraintPenalty * hardContstraintViolations
             
             violations=hardContstraintViolations
             
             #graph=self.gett_Value2(attr_fltList)[3]
             
             #j=[Max_Flow,violations]
             
             #f=max(j)-min(j)
             
             graph=self.get_sum_of_column_violaton(attr_fltList)[4]
             
             sum=self.get_sum_of_column_violaton(attr_fltList)[6]
             sum1=self.get_sum_of_column_violaton(attr_fltList)[10]
             df=self.get_sum_of_column_violaton(attr_fltList)[10]
             
             #violations=violations-min(j)/(f+1)
             #Max_Flow=Max_Flow-min(j)/(f+1)
             #max_flow=g.FordFulkerson(source, sink)
              
             
             return deviceViolations,-violations,sum_of_column_violaton,get_Variety_violation,-sum,df
             #return 100
         
         
    
################******MyGetValue****************########################################## 

    def print_output(self,attr_fltList):
        
        def list_duplicates(seq):
                tally = defaultdict(list)
                for i,item in enumerate(seq):
                     tally[item].append(i)
                return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>0)
        h=[]
        for dup in sorted(list_duplicates(attr_fltList)):
               h.append(dup)
           
           
        x=[]
        p=[]
           #y=[0 for _ in range(0,len(self.m))]
        y=[]
        type=list(self.type)
        for w,m in zip((range(0,len(h))),(range(0,len(type)))):
           #for w in (range(0,len(h))):
            
             l=[]
             j=[]
             g=[] 
        
             for c in h[w][1]:
            
                 l.append(self.items[c][0])
                 
                 s=self.items[c][int(type[w])]
######################################################################################                
                 
                 j.append(s)

#####################################################################################        
             p.append((h[w][0]+1,l,sum(j)))
             df=pd.DataFrame(p,columns=['NodeNumber','OperatorName','Speed'])
        
   
        return df
    
        
################*********************####################************************************        
    

            
    
        
@app.route('/main', methods=['GET','POST'])






     
# testing the class:
def main():
    
    json_ = request.json
    af=json_["OperatorsInformation"]
    items = list(af)
    df=json_["NodesInformation"]
    settings=json_["AccuracySettings"]
    Device=json_["DeviceSum"]
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
# create the desired optimization problem instace:
    TSP_NAME = "bayg29"  # name of problem
#tsp = tsp.TravelingSalesmanProblem(TSP_NAME)
    knapsack = Knapsack01Problem(af,df,Device,settings)
# Genetic Algorithm constants(settings and parameters):**************************************************
    POPULATION_SIZE = 2000
    MAX_GENERATIONS =100
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


    creator.create("FitnessMin", base.Fitness, weights=(1,))

# create the Individual class based on list of integers:
    creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)



    toolbox.register("randomOrder", create_number,len(knapsack),len(knapsack))



# create the individual creation operator to fill up an Individual instance with shuffled indices:
    toolbox.register("individualCreator", tools.initIterate, creator.Individual, toolbox.randomOrder)

# create the population creation operator to generate a list of individuals:
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# fitness calculation - compute the total distance of the list of cities represented by indices:
#def tpsDistance(individual):
    #return tsp.getTotalDistance(individual),  # return a tuple

    def evaluate(individual):
    
     
        return knapsack.GET_COST(individual)[1],
    #return knapsack.GET_COST(individual)[4],knapsack.GET_COST(individual)[1],
    toolbox.register("evaluate", evaluate)


# Genetic operators:
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1.0/len(knapsack))
#toolbox.register("mutate", tools.mutUniformInt,10,20, indpb=1.0/len(knapsack))


# Genetic Algorithm flow:


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

    best = tools.selBest(population, 1)[0]
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
            

    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])
   
    #print("-- Best Ever Fitness = ", hof.items[1])

    print("-- max_flow/violation = ",best.fitness.values[0])
       
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")
  
    
    print("DeviceCountViolaton:",knapsack.GET_COST(best)[0])
    
    #print("time is:",timeit.timeit(lambda: knapsack.GET_COST(best)[0], number=1))
    #print("Max_Flow:",knapsack.GET_COST(best)[1])
    print("SumOfRowColumnViolations:",knapsack.GET_COST(best)[2])
    
    #print("time is:",timeit.timeit(lambda: knapsack.GET_COST(best)[2], number=1))
    
    
    
    
    print("Variety_violation:",knapsack.GET_COST(best)[3])
    #print("time is:",timeit.timeit(lambda: knapsack.GET_COST(best)[3], number=1))
    
    
    print("violationS:",knapsack.GET_COST(best)[1])
    #print("time is:",timeit.timeit(lambda: knapsack.GET_COST(best)[1], number=1))
    
    
    
    print("sumrOW-min(Min_table):",knapsack.get_sum_of_column_violaton(best)[1])
    #print("time is:",timeit.timeit(lambda: knapsack.get_sum_of_column_violaton(best)[1], number=1))
    
    

    print("SumRowlistN:",knapsack.get_sum_of_column_violaton(best)[2])
    #print("time is:",timeit.timeit(lambda: knapsack.get_sum_of_column_violaton(best)[2], number=1))
    
    
    print("sumCol:",knapsack.get_sum_of_column_violaton(best)[3])
    #print("time is:",timeit.timeit(lambda: knapsack.get_sum_of_column_violaton(best)[3], number=1))
    
    
    #print("graph1:",knapsack.get_sum_of_column_violaton(best)[4])
    print("H:",knapsack.get_sum_of_column_violaton(best)[5])
    #print("time is:",timeit.timeit(lambda: knapsack.get_sum_of_column_violaton(best)[5], number=1))
    
    
    #print("Min_Table:",knapsack.get_sum_of_column_violaton(best)[6])
    #print("time is:",timeit.timeit(lambda: knapsack.get_sum_of_column_violaton(best)[6], number=1))
    #print("time is:",timeit.timeit(lambda: knapsack.get_sum_of_column_violaton(best)[9], number=1))
    print("outputx:",knapsack.get_sum_of_column_violaton(best)[9])
    print("UTPUT:",knapsack.print_output(best))
#****print output of optimization to output.csv file**************************
    knapsack.print_output(best).to_csv('output.csv')
    print("FinishTimeIs:",calculate_finish_time(best)[0],calculate_finish_time(best)[1],calculate_finish_time(best)[2],calculate_finish_time(best)[3],calculate_finish_time(best)[4])
    #print(len(knapsack.type))
    
    

    print("-- Best Ever Individual = ", best)
    print("-- Best Ever Fitness = ", best.fitness.values[0])



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
    #plt.show()

    
    
    
if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 13749 # If you don't provide any port the port will be set to 12345

    

    

    app.run(port=port, debug=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
  