# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 16:54:57 2021

@author: m.dorosti
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 10:24:40 2021

@author: m.dorosti
"""
from deap import base
from deap import creator
from deap import tools
import random
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import elitism
#import nurses
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
import traceback
import sys
from flask import Flask, request, render_template, session, redirect
from flask import Flask, redirect, url_for, request
import json
import pyodbc
from datetime import timedelta






app = Flask(__name__,template_folder='Template')

#app = Flask(__name__)
class NurseSchedulingProblem:
    """This class encapsulates the Nurse Scheduling problem
    """

    def __init__(self, StartDate,EndDate,hardConstraintPenalty,weeks,shiftPerDay,shiftsPerWeek,nurses,maxShiftsPerWeek,shiftMin,shiftMax,MaxBenefitOfEachGoods,MaxCapacityOfEachGOODS,DayCount,TedadTahvil):
        """
        :param hardConstraintPenalty: the penalty factor for a hard-constraint violation
        """
        # creating an empty list   
        self.hardConstraintPenalty = hardConstraintPenalty
        self.StartDate=StartDate
        self.EndDate=EndDate
        self.nurses=nurses
        self.shiftMin=shiftMin
        self.shiftMax=shiftMax
        self.maxShiftsPerWeek = maxShiftsPerWeek
        self.weeks=weeks
        self.shiftPerDay = shiftPerDay
        self.shiftsPerWeek= shiftsPerWeek
        self.MaxBenefitOfEachGoods=MaxBenefitOfEachGoods
        self.MaxCapacityOfEachGOODS=MaxCapacityOfEachGOODS
        self.DayCount=DayCount
        self.TedadTahvil=TedadTahvil
        '''self.countA=3
        self.countB=3
        self.countC=3
        self.A=50
        self.B=50
        self.C=50'''    
##################**********************########################************************************


    def __len__(self):
        """
        :return: the number of shifts in the schedule
        """
        return len(self.nurses) * self.shiftsPerWeek * self.weeks
    
    
##############********************###########******GetCostFunction*********##########****************#########################

    def getCost(self, schedule):
        """
        Calculates the total cost of the various violations in the given schedule
        ...
        :param schedule: a list of binary values describing the given schedule
        :return: the calculated cost
        """

        if len(schedule) != self.__len__():
            raise ValueError("size of schedule list should be equal to ", self.__len__())

        # convert entire schedule into a dictionary with a separate schedule for each nurse:
        nurseShiftsDict = self.getNurseShifts(schedule)
        shiftsPerWeekViolations = self.countShiftsPerWeekViolations(nurseShiftsDict)[1]
        nursesPerShiftViolations = self.countNursesPerShiftViolations(nurseShiftsDict)[1]
        CountShiftViolationForEachGood=self.CountShiftViolationForEachGood(nurseShiftsDict)
        countNumberViolation=self.CountNumber(nurseShiftsDict)
        # calculate the cost of the violations:
            
        hardContstraintViolations = countNumberViolation +nursesPerShiftViolations + shiftsPerWeekViolations+CountShiftViolationForEachGood
        
        return self.hardConstraintPenalty * hardContstraintViolations
    
        #return   ValueMaximization
    
### ##################******************************************************************
    def getNurseShifts(self, schedule):
        """
        Converts the entire schedule into a dictionary with a separate schedule for each nurse
        :param schedule: a list of binary values describing the given schedule
        :return: a dictionary with each nurse as a key and the corresponding shifts as the value
        """
        shiftsPerNurse = self.__len__() // len(self.nurses)
        nurseShiftsDict = {}
        shiftIndex = 0

        for nurse in self.nurses:
            nurseShiftsDict[nurse] = schedule[shiftIndex:shiftIndex + shiftsPerNurse]
            shiftIndex += shiftsPerNurse
        return nurseShiftsDict    
                    
##################3************MaximizingBenefitFunction********************************
    def getValue(self,nurseShiftsDict):
       p=[]
    
       PerShiftList = [shift for shift in zip(*nurseShiftsDict.values())]  
       PerShiftLis = [shift for shift in zip(nurseShiftsDict.values())]
       q=[]
       for i in PerShiftLis:
           q.append(sum(i[0]))
           
           
       #for j in range(len(q))    
       if q==self.maxShiftsPerWeek:
          
#PerShiftLis
           
       #if     
         for z in range(0,len(PerShiftList)):
                    for j in range(0,len(self.MaxBenefitOfEachGoods)):
            
                     #if sum(PerShiftList[j][z]) 
                        if PerShiftList[z][j]<= self.MaxCapacityOfEachGOODS[j]:
                   
        #totalValue=sum(PerShiftList[i]*self.MaxBenefitOfEachGoods[i])
                              p.append(PerShiftList[z][j]*self.MaxBenefitOfEachGoods[j])
      
                                  
         u=sum(p)
       
       else:
           
           
         u=0
       
       
       #k=(-u)
       return -u    
##############################***countConsecutiveShiftViolations####### ###############  
    
  #مورد استفاده ما نیست  

    def countConsecutiveShiftViolations(self, nurseShiftsDict):
        """
        Counts the consecutive shift violations in the schedule
        :param nurseShiftsDict: a dictionary with a separate schedule for each nurse
        :return: count of violations found
        """
        violations = 0
        # iterate over the shifts of each nurse:
        for nurseShifts in nurseShiftsDict.values():
            # look for two cosecutive '1's:
            for shift1, shift2 in zip(nurseShifts, nurseShifts[1:]):
                if shift1 == 1 and shift2 == 1:
                    violations += 1
        return violations   
##########*****************################**********CountShiftViolationForEachGood###############################


    def CountShiftViolationForEachGood(self,nurseShiftsDict):
    
       violations=0
       G=[10,14,36,56,70]
    
       total = [shift for shift in nurseShiftsDict.values()]
    
       for i in range (len(total)):
        
         for j in range(len(total[i])):
          
           if total[i][j]>self.MaxCapacityOfEachGOODS[i]:
           
              violations=violations+(total[i][j]-self.MaxCapacityOfEachGOODS[i])
           
        
       return violations    
 #######******#######******####################CountShiftsPerWeekViolations###################################  
#جمع شیفت ها برای هر محصول کمتر از میزان سفارش مورد نظر نباشد    

    def countShiftsPerWeekViolations(self, nurseShiftsDict):
        """
        Counts the max-shifts-per-week violations in the schedule
        :param nurseShiftsDict: a dictionary with a separate schedule for each nurse
        :return: count of violations found
        """
        
        violations = 0
        weeklyShiftsList = []
        
        
        # iterate over the shifts of each nurse:
        
        for nurseShifts in nurseShiftsDict.values():  # all shifts of a single nurse
            # iterate over the shifts of each weeks:
            for i in range(0, self.weeks * self.shiftsPerWeek, self.weeks*self.shiftsPerWeek):
                # count all the '1's over the week:
                weeklyShifts = sum(nurseShifts)
                weeklyShiftsList.append(weeklyShifts)
            for nursIndex, numOfshifts in enumerate(weeklyShiftsList):    
            
                  if  numOfshifts < self.maxShiftsPerWeek[nursIndex]:
                     
                     violations +=self.maxShiftsPerWeek[nursIndex]-numOfshifts
                  elif numOfshifts > self.maxShiftsPerWeek[nursIndex] :
                     
                     violations +=numOfshifts-self.maxShiftsPerWeek[nursIndex]
                  
                     #violations =violations +1
                  

        return weeklyShiftsList, violations
          
#3weeklyShiftsList, violations=countShiftsPerWeekViolations(self, nurseShiftsDict)
#################*****محدودیت تعداد تحویل سفارش*****************************************************************
    def CountNumber(self,nurseShiftsDict):
         violations=0
         a=0
         b=0
         c=0
         #در روز ان ام باید این تعداد تحویل داده شود.        
         #self.DayCount=[4,4,4,4,3]
         #self.TedadTahvil=[50,50,50,50,30]
         violation=0
         viola=[]
         
         a=[]

         for i in range(0,len(nurseShiftsDict)):
   
              sum=0 
        
              for j in range(0,self.DayCount[i]):
        
                   sum=sum+nurseShiftsDict[list(nurseShiftsDict.keys())[i]][j]
              a.append(sum)
              
              for i in range(0,len(a)):
                  
                  if a[i]<self.TedadTahvil[i]:
                      
                      violation=violation+(self.TedadTahvil[i]-a[i])
                      #viola.append(violation)
              violations=violation
         return violations
#################***********maxBenefit##########################################################********************8

#رعایت سریع تر تولید شدن بیشترین سود'''
    def countMaxBenefitViolations(self, nurseShiftsDict):
        """
        Counts the number-of-nurses-per-shift violations in the schedule
        :param nurseShiftsDict: a dictionary with a separate schedule for each nurse
        :return: count of violations found
        """
        g=[]
        #violations=0
        for num1, num2 in zip(self.maxShiftsPerWeek,self.MaxBenefitOfEachGoods):
            
            g.append(num1*num2)
       
        inds = np.array(g).argsort()
        sorted=np.array(inds)[::-1]
        violations=0
       # a=[]
        PerShiftList = [shift for shift in zip(*nurseShiftsDict.values())]
       
        for i in range(0,len(PerShiftList)):
            
            f=np.array(PerShiftList[i])
           
            array1=np.argsort(f)
            m=[]
            for s in range (0,len(self.maxShiftsPerWeek)):
                m.append("m")
            
    
            if (array1==inds).all()==False:
                
                
                for n, j in zip(inds, range(len(f))):
    
                    m[n]=f[array1[j]]            
         
 #####################################**countNursesPerShiftViolations***###############################   
    
  #رعایت ظرفیت تولید در هر شیفت  

    def countNursesPerShiftViolations(self, nurseShiftsDict):
        """
        Counts the number-of-nurses-per-shift violations in the schedule
        :param nurseShiftsDict: a dictionary with a separate schedule for each nurse
        :return: count of violations found
        """
        # sum the shifts over all nurses:
        totalPerShiftList = [sum(shift) for shift in zip(*nurseShiftsDict.values())]

        violations = 0
        # iterate over all shifts and count violations:
        for shiftIndex, numOfNurses in enumerate(totalPerShiftList):
            #dailyShiftIndex = shiftIndex % self.shiftPerDay
            dailyShiftIndex = shiftIndex
            #dailyShiftIndex = shiftIndex # -> 0, 1, or 2 for the 3 shifts per day
            if (numOfNurses > self.shiftMax[dailyShiftIndex]):
                violations += numOfNurses - self.shiftMax[dailyShiftIndex]
            elif (numOfNurses < self.shiftMin[dailyShiftIndex]):
                violations += self.shiftMin[dailyShiftIndex] - numOfNurses

        return totalPerShiftList, violations   
#########################countShiftPreferenceViolations###########################################3
   
            
################################**PRINT SCADULING INFORMATION**###############################################################33    
    def printScheduleInfo(self, schedule):
        """
        Prints the schedule and violations details
        :param schedule: a list of binary values describing the given schedule
        """
        nurseShiftsDict = self.getNurseShifts(schedule)
        p=[20,30,15]
        print("Schedule for each nurse:")
        for nurse in nurseShiftsDict:
        #for i in nurseShiftsDict:
            #for j in range(0,len(p)):
            # all shifts of a single nurse
            print(nurse, ":", nurseShiftsDict[nurse])
            #print( ":", nurseShiftsDict[i])
            for j in range(0,len(nurseShiftsDict[nurse])):
                #for j in range (0,3):
                   print(nurseShiftsDict[nurse][j]) 
                    #print("مقدار پارچه مورد نیاز",nurseShiftsDict[nurse][i])
       

        print("consecutive shift violations = ", self.countConsecutiveShiftViolations(nurseShiftsDict))
        print()

        weeklyShiftsList, violations = self.countShiftsPerWeekViolations(nurseShiftsDict)
        print("weekly Shifts = ", weeklyShiftsList)
        print("Shifts Per Week Violations = ", violations)
        print()
        a="weekly Shifts="+str(weeklyShiftsList)
        totalPerShiftList, violations = self.countNursesPerShiftViolations(nurseShiftsDict)
        print("Nurses Per Shift = ", totalPerShiftList)
        print("Nurses Per Shift Violations = ", violations)
        print()
        b="Shifts Per Week Violations =" + str(violations)
        
        #c="Nurses Per Shift = "+str(totalPerShiftList)
        c="totalPerShiftList="+str(violations)
        d="Nurses Per Shift Violations =" + str(violations)
        #shiftPreferenceViolations = self.countShiftPreferenceViolations(nurseShiftsDict)
        #print("Shift Preference Violations = ", shiftPreferenceViolations)
        print()
        q=[]
        #a=print("consecutive shift violations = ", self.countConsecutiveShiftViolations(nurseShiftsDict))
        print()
       
        weeklyShiftsList, violations = self.countShiftsPerWeekViolations(nurseShiftsDict)
        #b=print("weekly Shifts = ", weeklyShiftsList)
        print("Shifts Per Week Violations = ", violations)
        print()
        a="weekly Shifts="+str(weeklyShiftsList)
        b="Shifts Per Week Violations =" + str(violations)
        
        totalPerShiftList, violations = self.countNursesPerShiftViolations(nurseShiftsDict)
        c="Producted Per Shift = "+str(totalPerShiftList)
        d="Producted Per Shift Violations =" + str(violations)
        
        import pandas as pd
        
        violations =self.CountShiftViolationForEachGood(nurseShiftsDict)
        n="CountShiftViolationForEachGood =" + str(violations)
        
        violations=self.CountNumber(nurseShiftsDict)
        l="TedadSefareshViolation =" + str(violations)
        
        
        
        g=nurseShiftsDict
        
        print()
        q.append(a)
        q.append(b)
        q.append(c)
        q.append(d)
        q.append(n)
        q.append(g)
        q.append(l)
        
        print()
        return a,b,c,d,weeklyShiftsList,totalPerShiftList,nurseShiftsDict,q,g
        
#########################8***************************#######********************##################################
# -*- coding: utf-8 -*-


@app.route('/main', methods=['GET','POST'])     
# Genetic Algorithm flow:
def main():
    
    HARD_CONSTRAINT_PENALTY = 80
    # the penalty factor for a hard-constraint violation

    # Genetic Algorithm constants:
    POPULATION_SIZE = 500
    P_CROSSOVER = 0.9  # probability for crossover
    P_MUTATION = 0.2   # probability for mutating an individual
    MAX_GENERATIONS = 3000
    HALL_OF_FAME_SIZE = 100

    # set the random seed:
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    import pandas as pd
    json_ = request.json  
    weeks=json_["weeks"] 
    shiftPerDay=json_["shiftPerDay"]
    shiftsPerWeek=json_["shiftsPerWeek"]
    maxShiftsPerWeek=json_["maxShiftsPerWeek"]
    shiftMin=json_["shiftMin"]
    shiftMax=json_["shiftMax"]
    nurses=list(json_["NameOfGoods"])
    MaxBenefitOfEachGoods=json_["MaxBenefitOfEachGoods"]
    MaxCapacityOfEachGOODS=json_["MaxCapacityOfEachGOODS"]
    DayCount=json_["DayCount"]
    TedadTahvil=json_["TedadTahvil"]
    StartDate=json_["StartDate"]
    EndDate=json_["EndDate"]
    
    
    '''weeks=request.form.get("weeks")
    shiftPerDay = request.form.get("shiftPerDay")
    shiftsPerWeek = request.form.get("shiftsPerWeek")
    maxShiftsPerWeek = request.form.get("maxShiftsPerWeek")
    shiftMin = request.form.get("shiftMin")
    shiftMax = request.form.get("shiftMax")
    #NameOfGoods = request.form.get("NameOfGoods")
    nurses = request.form.get("NameOfGoods")
    MaxBenefitOfEachGoods=request.form.get("MaxBenefitOfEachGoods")'''
    
    toolbox = base.Toolbox()
    
    # create the nurse scheduling problem instance to be used:
    nsp = NurseSchedulingProblem(StartDate,EndDate,HARD_CONSTRAINT_PENALTY,weeks=weeks,shiftPerDay=shiftPerDay,shiftsPerWeek=shiftsPerWeek,nurses=nurses,maxShiftsPerWeek= maxShiftsPerWeek,shiftMin=shiftMin,shiftMax=shiftMax,MaxBenefitOfEachGoods=MaxBenefitOfEachGoods,MaxCapacityOfEachGOODS=MaxCapacityOfEachGOODS,DayCount=DayCount,TedadTahvil=TedadTahvil)

    # define a single objective, maximizing fitness strategy:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # create the Individual class based on list:
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # create an operator that randomly returns 0 or 1:
    #toolbox.register("zeroOrOne", random.randint, 0, 1)
    toolbox.register("attr_flt", random.randint,0, 50)
    # create the individual operator to fill up an Individual instance:
    #toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(nsp))
    toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.attr_flt, len(nsp))

    # create the population operator to generate a list of individuals:
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


    # fitness calculation
    def getCost(individual):
       return nsp.getCost(individual), 
       #return len(nsp),# return a tuple


    toolbox.register("evaluate", getCost)

    # genetic operators:
    toolbox.register("select", tools.selTournament, tournsize=2)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/len(nsp))
    #weeks = request.args.get('weeks')
    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = elitism.eaSimpleWithElitism(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

    # print best solution found:
    best = hof.items[0]
    print("-- Best Individual = ", best)
    print("-- Best Fitness = ", best.fitness.values[0])
    print()
    print("-- Schedule = ")
    f=print(nsp.printScheduleInfo(best))
    #print(f)

    # extract statistics:
    minFitnessValues, meanFitnessValues = logbook.select("min", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(minFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Min / Average Fitness')
    plt.title('Min and Average fitness over Generations')
    plt.show()
    
    
    a,b,c,d,weeklyShiftsList,totalPerShiftList,nurseShiftsDict,q,g=nsp.printScheduleInfo(best)
    conn=pyodbc.connect('Driver={Sql Server};'
                                   'Server=192.168.100.17\\SQL2019;'
                                   'Database=Modern_Master;'
                                  'UID=sa;'
                                   'PWD=PAYA+master;'
                                   )
    
    cursor = conn.cursor()
    cursor.execute("DELETE FROM  bi.ManufacturePlanning ;")
    date=datetime.now()
    #date=str(StartDate[0])+"-"+str(StartDate[1])+"-"+str(StartDate[2])
    EndDate=str(EndDate[0])+"-"+str(EndDate[1])+"-"+str(EndDate[2])
    #for i in range (0,len(best)):
        #for j in nurseShiftsDict.keys():
        #for j in range (0,len( nurseShiftsDict.values()()))    
    gdsAmount=list(nurseShiftsDict.values())
    gdsName=list(nurseShiftsDict.keys())
    datee=[]
    for i in range(0,shiftsPerWeek):
        date += timedelta(days=1)
        datee.append(date.strftime("%Y-%m-%d"))
    #date=['one','two','three','four','five']*shiftsPerWeek
    date=datee*shiftsPerWeek
    IdGdss=[]        
    for i in range (0,len(gdsAmount)):
      for j in range (0,len(gdsAmount[i])):
        IdGdss.append(gdsName[i])
    
    
    for i in range (0,len(best)):
        
                  #date += timedelta(days=1)
                  cursor.execute("INSERT INTO  bi.ManufacturePlanning (IdGds,Date,Amount)  VALUES (?,?,?);"  , IdGdss[i],date[i],best[i])
                  conn.commit()

    return str(q)

@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      #return render_template('login.html')
      #user = request.form['nm']
      
      weeks=request.form.get('weeks',type=int)
      DayCount=request.form.get('DayCount',type=int)
      TedadTahvil=request.form.get('TedadTahvil',type=int)
     # weeks=int(weeks)
      shiftPerDay=request.form.get('shiftPerDay',type=int)
      #shiftPerDay=int(shiftPerDay)
      
      shiftsPerWeek=request.form.get('shiftsPerWeek',type=int)
      #shiftsPerWeek=int(shiftsPerWeek)
      
      
      maxShiftsPerWeek=request.form.getlist('maxShiftsPerWeek',type=int)
      shiftMin=request.form.getlist('shiftMin',type=int)
      shiftMax=request.form.getlist('shiftMax',type=int)
      nurses=request.form.getlist('NameOfGoods',type=int)
      MaxBenefitOfEachGoods=request.form.getlist('MaxBenefitOfEachGoods',type=int)
      MaxCapacityOfEachGOODS=request.form.getlist('MaxCapacityOfEachGOODS',type=int)
      
      return redirect(url_for('main',weeks=weeks,shiftPerDay=shiftPerDay, shiftsPerWeek= shiftsPerWeek,maxShiftsPerWeek=maxShiftsPerWeek, shiftMin=shiftMin,shiftMax=shiftMax,nurses=nurses,MaxBenefitOfEachGoods=MaxBenefitOfEachGoods, MaxCapacityOfEachGOODS=MaxCapacityOfEachGOODS))
      
      
      #return redirect(url_for('success',name = user))
   #else:
      #user = request.args.get('nm')
     # return redirect(url_for('success',name = user))
    
    


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 13745 # If you don't provide any port the port will be set to 12345

    

    

    app.run(port=port, debug=True)











    

