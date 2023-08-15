# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 11:44:02 2022

@author: m.dorosti
"""
import pandas as pd
import numpy as np 
import networkx as nx

df=pd.read_excel('input.xlsx','NodesInformation')

NumberOfNodes=len(df.nodei.unique())+1
    
z= [[0 for _ in range(NumberOfNodes)] for _ in range(NumberOfNodes)]

G = nx.DiGraph()
                  #G.add_edge("A", "B", capacity=self.items[attr_fltList[0]][1])
                  #G.add_edge("A", "C", capacity=self.items[attr_fltList[1]][2])
G.add_edge("A", "B", capacity=15)
G.add_edge("A", "C", capacity=15)
#G.add_edge("C", "D", capacity=self.items[attr_fltList[2]][3])
G.add_edge("C", "D", capacity=13)
G.add_edge("B", "E", capacity=14)
G.add_edge("E", "F", capacity=12)
G.add_edge("D", "F", capacity=10)
G.add_edge("F", "G", capacity=9)
G.add_edge("G", "0", capacity=7)
                  
f=nx.maximum_flow(G, "A", "0")[1]
s=nx.minimum_cut(G,"A","0")
g=nx.minimum_cut_value(G,"A","0")
print("f is:",f)
print("g is: ",g)
print('s is:',s)




