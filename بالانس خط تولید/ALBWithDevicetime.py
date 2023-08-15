import numpy as np

import statistics

#import networkx as nx

#import GraphClass
#import GraphClass2
import time
from statistics import mean

from collections import defaultdict
import pandas as pd
import random
from joblib import Parallel ,delayed

class Knapsack01Problem:
    """This class encapsulates the Knapsack 0-1 Problem from RosettaCode.org
    """

    def __init__(self):

        # initialize instance variables:
        self.items = []
        self.maxCapacity = 0

        # initialize the data:
        self.__initData()

    def __len__(self):
        """
        :return: the total number of items defined in the problem
        """
        return len(self.items)
        #return 1
        #return self.NumberOfEdges

    def __initData(self):
        """initializes the RosettaCode.org knapsack 0-1 problem data
        
        """
        
        self.af=pd.read_excel('input.xlsx','OperatorsInformation')
        #self.af=pd.read_excel('a.xlsx','OperatorsInformation')
        
        #self.af=pd.read_excel('a.xlsx','OperatorsInformation')
        self.af=self.af[0:70]
        self.records = self.af.to_records(index=False)
        self.items = list(self.records)
        
        self.InputTime=60
        self.NumberOfgoods=400
        
        
        
        '''self.items = [
             
            ("ALI", 2, 2,5,5,5,5,0),
            ("REZA", 4, 4,6,6,6,6,0),
            ("NASER", 2, 2,10,2,4,4,0),
            ("AHMAD", 3, 3,4,4,4,4,0),
            ("NADER", 7, 7,4,4,6,6,0),
            ("JAVAD", 2, 2,10,5,3,3,0),
            ("DAVOOD", 4, 4,10,6,5,5,0),
            ("AMIR", 5, 5,2,2,5,5,0),
            ("SAEID", 3, 3,4,4,6,6,0),
            ("SOHRAB", 3, 3,4,4,6,6,0),
            ("JAFAR", 2, 2,5,5,8,8,0),
            ("KAMRAN", 4, 4,10,8,5,5,0),
            ("KEYVAN", 5, 5,2,2,5,5,0),
            ("KAMYAR", 7, 7,2,20,4,40,0),
            ("NIMA", 6, 6,4,4,8,6,0),
            ("BAHRAM", 2, 2,3,3,1,10,0),
            ("FARHAD", 4, 4,2,2,5,5,0),
            ("FARHOOD", 2, 2,2,2,4,4,0),
            ("AMIRREZA", 3, 3,10,4,6,6,0),
            ("HADI", 7, 7,2,2,6,6,0),
            ("SAM", 2, 2,5,5,1,1,0),
            ("MEHRAN", 10, 4,10,2,5,5,0),
            ("MEHRDAD", 1, 1,2,2,5,5,0),
            ("MEHDI", 3, 3,4,4,6,6,0),
            ("PEYMAN", 4, 4,5,5,2,2,0)]'''
            
            
        
        
        self.df=pd.read_excel('input.xlsx','NodesInformation')
        #self.df=pd.read_excel('a.xlsx','NodesInformation')
        self.df=self.df.iloc[0:55]
        #self.df=self.df.iloc[0:13]
        
        #self.df=pd.read_excel('a.xlsx','NodesInformation')
        #self.df=self.df.iloc[0:16]
        k=self.df.loc[self.df['nodei'] != 0]
        
        self.m=list(k.iloc[:,0])
        
        self.n=list(k.iloc[:,1])
        
        #self.e=self.df.loc[self.df['nodei'] == 0]
        self.e=self.df
        self.f=list(self.e.iloc[:,0])
        self.g=list(self.e.iloc[:,1])
        
        
        
        self.Device=pd.read_excel('input.xlsx','DeviceSum')
        #self.Device=pd.read_excel('a.xlsx','DeviceSum')
        self.device=list(self.Device.iloc[:,1])
       
        
        
       
        
        self.type=k.iloc[:,3]
        
        
        
        
        
        self.time=[16,33,32,19,43,31,0,45,26,70,42,107,
                   63,117,54,23,63,33,45,60,36,90,110,81,45,45,40,33,60,90,45,48,73,111,111,0,58,86,51
                   ,130,85,122,80,99,54,193,80,125,100,100,93,80,60,48,54,115,54,93,39,1,1,56,1,1,1] 
        
        
        
        self.hardConstraintPenalty=10
        
        self.NumberOfNodes=len(self.df.nodei.unique())+1
    
        self.z= [[0 for _ in range(self.NumberOfNodes)] for _ in range(self.NumberOfNodes)]
        
        self.settings=pd.read_excel('input.xlsx','AccuracySettings')
        
        self.accuracy=self.settings.iloc[0,0]
        
        self.sie=self.settings.iloc[0,1]
        self.sae=self.settings.iloc[0,2]
        
        W=self.df.loc[self.df['nodei'] == 0]
        self.W=list(W)
        
        
###################################################################################################       
       
    #def measure_time(f):

         #def timed(*args, **kw):
              #ts = time.time()
              #result = f(*args, **kw)
              #te = time.time()

              #print ('%r (%r, %r) %2.2f sec' % \
                         #(f.__name__, args, kw, te-ts))
              #return result

   #return timed    
       
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
       #a_set = set(attr_fltList)
       #for i in range(0,len(attr_fltList)):
        #if attr_fltList[i] > self.NumberOfNodes-3:
            #attr_fltList[i]=int(np.random.randint(self.NumberOfNodes-3, size=1))
       
       
         violations=0
       #a_set = frozenset(attr_fltList)
       #if len(a_set)==len(self.type):
       #if len(a_set)==len(self.type):
         #violations=0
         #attr_fltList[0]
           
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
                    
                    
                    
         #if len(F)!=len(hh):
            #F=F[0:len(hh)] 
            
            
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
             #elif p<o:
                 #violations=violations+(o-p)
       #else:
         
         #violations=violations+1
         return violations           
         
         
################***************get sum of column violaton########################################
       
    def get_sum_of_column_violaton(self,attr_fltList):
        
       #for i in range(0,len(attr_fltList)):
           #if attr_fltList[i] > 46:
             #attr_fltList[i]=int(np.random.randint(46, size=1))

           z=self.z
       #violations=0
       #z=self.z
       #z = [[0 for _ in range(65)] for _ in range(65)]
       #attr_fltList=np.array(attr_fltList)
       
       #a_set = set(attr_fltList)
       #if len(a_set)<=len(self.type): 
       #if len(attr_fltList[0])==len(self.type):     
           #source=[5, 1, 5, 0, 2, 4, 3, 4, 5, 1, 1, 1, 1, 3, 0, 1, 5, 2, 3, 5, 5, 1, 2, 1, 5]
           
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
           #p=[]
           #y=[0 for _ in range(0,len(self.m))]
           y=[]
           type=list(self.type)
           for w,m in zip((range(0,len(h))),(range(0,len(type)))):
           #for w in (range(0,len(h))):
            
             #l=[]
             j=[]
             g=[] 
        
             for c in h[w][1]:
            
                 #l.append(self.items[c][0])
                 
                 s=self.items[c][int(type[w])]
######################################################################################                
                 
                 j.append(s)
             #if self.time[w]>90:
                 #x.append((h[w][0],l,sum(j)))
               
             #else:
                 #x.append((h[w][0],l,max(j)))  
#####################################################################################        
             #p.append((h[w][0],l,sum(j)))
             #df=pd.DataFrame(p,columns=['NodeNumber','OperatorName','Speed'])
             x.append(sum(j[1:]))
           #x=[]
           
           #x.append(5)
             
           #for i in range(0,len(x)):
                  #y.append(x[i][2])
                  
             #for k,f in zip(h[w][0],range(0,len(x))):
             #y[h[w][0]]=sum(j)
##########################################################################################                  
             
             #G=len(y)
             #if len(y)<len(self.m):
                # for i in range(0,(len(self.m)-len(y))):
                     #y.append(5)
             #if len(self.m)>=len(y):
                 
           #x=[] 
           #y=[]
           #for  s, p,v in zip(self.m,self.n,range(0,len(y))):
           for  s, p,v in zip(self.m,self.n,range(0,len(x))):
           #for  s, p in zip(self.m,self.n):
                 
                 #if s in y:
                 
                 #if len(self.m)<=len(y):
    
                     #z[s][p]=y[v]
                     z[s][p]=x[v]
                     #z[s][p]=2
                     
                 #else:
                     #z[s][p]=random.randint(1,9)
                     
                     
             #for  s, p,v in zip(self.m,self.n,range(len(y),len(s)):
                 
                  #z[s][p]=y[v]
             #else:
               #for  s, p in zip(self.m,self.n) : 
                     #z[s][p]=1
             
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
                 
                 #sumRow = sumRow + graph[i][j] 
                 RowList.append(graph1[i][j])
                 ColumnList.append(graph1[j][i])
                 #sumCol=sumCol+graph1[j][i]
                 sumCol=sumCol+graph1[j][i]
                 sumRow=sumRow+graph1[i][j]
                 #SumRow.append(sumRow)
                 #SumRow=SumRow+graph1[i][j]
                 
             S.append(ColumnList)
             N.append(sumRow)      
             
              
             
           p=[k for k, e in enumerate(graph1[0]) if e != 0] 
           
          
                   
           H=[]
           for r in range(0,len(S)):
               h=[k for k, e in enumerate(S[r]) if e != 0] 
               H.append(h)
           
           
           
           #Min_table = [0 for _ in range(len(p))]
           #K= [0 for _ in range(len(p))]
           
           
           #for P , i in zip(p,range(0,len(Min_table))):
               
                  #Min_table[i]=graph1[0][P]
                  #K[i]=N[P]
                  
                  
          
           F = [0 for _ in range(len(N))]
           k=[0 for _ in range(len(N))]
           '''A=[]
           for P,i in zip(p,range(0,len(Min_table))):
              #F=[] 
              j=N[P-1]-Min_table[i]
              F[P-1]=abs(j)
              A.append(abs(j))
              if abs(j)>3:
                          #violations=violations+abs(J)
                          violations=violations+max(F)
              
              
              #F[.append(abs(j))
           B=[]   
           for  i  in range(0,len(N)):
               for P in p:
               
                  if i!= P-1:
                   
                      J=N[i]-min(graph1[b][i+1] for b in H[i] )
                      
                      F[i]=abs(J)
                      B.append(abs(J))
                      if abs(J)>3:
                          #violations=violations+abs(J)
                          violations=violations+max(F)
              
                      #F.append(abs(j))'''
              
              
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
                          
                           #violations=violations+1  
              
              
              #'''for i in range(len(p),len(N)) :
               
              #j=N[i]-min(graph1[b][i+1] for b in H[i] )
              
             # F.appen☺d(abs(j))'''
              
           #g=sum(A)+sum(B) 
           T=[]
           #for P in p:
           for i in range(0,len(N)):    
               for P in p:
                   if i== P-1:
                       T.append(F[i])
           '''def Appendfunc(i,P):
               if i== P-1:
                  T.append(F[i])
               return(T)
                       
           Parallel(n_jobs=2)(delayed(Appendfunc)(i,P)for i in range (0,len(N)) for P in p  )'''           
           #violation=0            
           #for i in range(0,len(T)) :
               #if abs(T[i])>5 :
                   #violation=violation+2
                       
           c=sum(F)
           g=sum(T)
           df=1
           #if c>150:
           if c>self.sae:
               violations=violations+1
               violations=violations+(c-self.sae)
           #for s in range(0,len(F)):
               #if F[s]<3:
           #if g>15:
           if g>self.sie:
               #violations=violations+1
               violations=violations+(g-self.sae)
                   #violations=violations+max(F)
                   #violations=violations+1
                   #violations=0
                   #violations=violations+10
              # else:
                  # violations=violations+1
       #else:
           
           #violations=50
           #F=0
           #N=[]
           #sumCol=0
           #graph1=0
           #H=0
           #Min_table=0
           #g=0
           #S=0
            
           return violations,F,N,sumCol,graph1,H,g,S,x,y,c,df
           
                   
                   

             
             
           
#########################################################################################



    def get_Variety_violation(self,attr_fltList):
        
        #for i in range(0,len(attr_fltList)):
           #if attr_fltList[i] > self.NumberOfNodes-3:
            #attr_fltList[i]=int(np.random.randint(self.NumberOfNodes-3, size=1))
        violations=0
        a_set = set(attr_fltList)
        #a_set = frozenset(tuple(attr_fltList[0]))
        #if len(a_set)==len(self.type):
        if len(a_set)==len(self.type):
        #if len(a_set)==len(self.type):
        #if len(attr_fltList)==len(self.type): 
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
             #if self.time[w]>90:
                 #x.append((h[w][0],l,sum(j)))
               
             #else:
                 #x.append((h[w][0],l,max(j)))  
#####################################################################################        
             p.append((h[w][0],l,sum(j)))
             df=pd.DataFrame(p,columns=['NodeNumber','OperatorName','Speed'])
        
   
        return df
    '''def gett_Value(self,attr_fltList):
        
        
         
    
        
        for i in range(0,len(attr_fltList)):
             a_set = set(attr_fltList)
             if len(attr_fltList) == len(a_set):
                 
                   
                   
                 
                  G = nx.DiGraph()
                  #G.add_edge("A", "B", capacity=self.items[attr_fltList[0]][1])
                  #G.add_edge("A", "C", capacity=self.items[attr_fltList[1]][2])
                  G.add_edge("A", "B", capacity=50)
                  G.add_edge("A", "C", capacity=50)
                  G.add_edge("C", "D", capacity=self.items[attr_fltList[2]][3])
                  G.add_edge("B", "E", capacity=self.items[attr_fltList[3]][4])
                  G.add_edge("E", "F", capacity=self.items[attr_fltList[4]][5])
                  G.add_edge("D", "F", capacity=self.items[attr_fltList[5]][6])
                  
                  f=nx.maximum_flow(G, "A", "F")[0]
           
             else:
              
                  f=0 
             
        #print(nx.maximum_flow(G, "A", "J")[1])
        #print(nx.maximum_flow(G,"A","J")[0])
             return(f)'''
        
################*********************####################************************************        
    

            
    
        

# testing the class:
def main():
    # create a problem instance:
    knapsack = Knapsack01Problem()

    # creaete a random solution and evaluate it:
    randomSolution = np.random.randint(2, size=len(knapsack))
    size=len(knapsack)
    print(size)
    print("Random Solution = ")
    print(randomSolution)
    
    #knapsack.printItems(randomSolution)


if __name__ == "__main__":
    main()