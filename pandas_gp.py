# -*- coding: utf-8 -*-
"""
Standard GP Algorithm (Strongly typed)

Created on Sun Aug 21 10:40:14 2016

@author: jm
"""

import operator
import math
import random
import cPickle
import math

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx


from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

#from scoop import futures
#import multiprocessing

primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,
          97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,
          181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,
          277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,
          383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,
          487,491,499] #,503,509,521,523,541,547,557,563,569,571,577,587,593,599,
#          601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,
#          709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,
#          827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,
#          947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,
#          1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,
#          1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,
#          1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,
#          1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,
#          1439,1447,1451,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511,
#          1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,
#          1609,1613,1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,
#          1709,1721,1723,1733,1741,1747,1753,1759,1777,1783,1787,1789,1801,
#          1811,1823,1831,1847,1861,1867,1871,1873,1877,1879,1889,1901,1907,
#          1913,1931,1933,1949,1951,1973,1979,1987,1993,1997,1999,2003,2011,
#          2017,2027,2029,2039,2053,2063,2069,2081,2083,2087,2089,2099,2111,
#          2113,2129,2131,2137,2141,2143,2153,2161,2179,2203,2207,2213,2221,
#          2237,2239,2243,2251,2267,2269,2273,2281,2287,2293,2297,2309,2311,
#          2333,2339,2341,2347,2351,2357,2371,2377,2381,2383,2389,2393,2399,
#          2411,2417,2423,2437,2441,2447,2459,2467,2473,2477,2503,2521,2531,
#          2539,2543,2549,2551,2557,2579,2591,2593,2609,2617,2621,2633,2647,
#          2657,2659,2663,2671,2677,2683,2687,2689,2693,2699,2707,2711,2713,
#          2719,2729,2731,2741,2749,2753,2767,2777,2789,2791,2797,2801,2803,
#          2819,2833,2837,2843,2851,2857,2861,2879,2887,2897,2903,2909,2917,
#          2927,2939,2953,2957,2963,2969,2971,2999,3001,3011,3019,3023,3037,
#          3041,3049,3061,3067,3079,3083,3089,3109,3119,3121,3137,3163,3167,
#          3169,3181,3187,3191,3203,3209,3217,3221,3229,3251,3253,3257,3259,
#          3271,3299,3301,3307,3313,3319,3323,3329,3331,3343,3347,3359,3361,
#          3371,3373,3389,3391,3407,3413,3433,3449,3457,3461,3463,3467,3469,
#          3491,3499,3511,3517,3527,3529,3533,3539,3541,3547,3557,3559,3571,
#          3581,3583,3593,3607,3613,3617,3623,3631,3637,3643,3659,3671,3673,
#          3677,3691,3697,3701,3709,3719,3727,3733,3739,3761,3767,3769,3779,
#          3793,3797,3803,3821,3823,3833,3847,3851,3853,3863,3877,3881,3889,
#          3907,3911,3917,3919,3923,3929,3931,3943,3947,3967,3989,4001,4003,
#          4007,4013,4019,4021,4027,4049,4051,4057,4073,4079,4091,4093,4099,
#          4111,4127,4129,4133,4139,4153,4157,4159,4177,4201,4211,4217,4219,
#          4229,4231]
          
          
# load the data
ibex = cPickle.load(open("ibex.pickle", "rb"))

#we need to have pd_df_bool terminals for gp to work
#better to have them as a column in the dataset
#then to use a terminal calling np.ones or np.zeros
#as we are not constrained by the size of the vector

ibex["Ones"] = True
ibex["Zeros"] = False


# Transaction costs - 10 Eur per contract
#cost = 30
cost = 50 # equiv (30/10000) * 10 points

# Split into train and test
train = ibex["2000":"2015"].copy()
test = ibex["2016":].copy()

# Functions and terminal for GP

class pd_df_float(object):    
    pass

class pd_df_bool(object):
    pass

def f_gt(df_in, f_value):
    return df_in > f_value

def f_lt(df_in, f_value):
    return df_in < f_value

def protectedDiv(left, right):
    try: return left / right
    except ZeroDivisionError: return 0.0
        
def pd_add(left, right):
    return left + right
    
def pd_subtract(left, right):
    return left - right
      
def pd_multiply(left, right):
    return left * right

def pd_divide(left, right):
    return left / right
    
def pd_diff(df_in, _periods):
    return df_in.diff(periods=abs(_periods))

def sma(df_in, periods):
    return pd.rolling_mean(df_in, abs(periods))

def ewma(df_in, periods):
    if abs(periods) < 2:
        return df_in
    else:
        return pd.ewma(df_in, abs(periods), min_periods=abs(periods)) 
    
def hh(df_in, periods):
    if abs(periods) < 2:
        return df_in
    else:
        return pd.rolling_max(df_in, abs(periods), min_periods=abs(periods))

def ll(df_in, periods):
    if abs(periods) < 2:
        return df_in
    else:
        return pd.rolling_min(df_in, abs(periods), min_periods=abs(periods))

def pd_std(df_in, periods):
    if abs(periods) < 2:
        return df_in
    else:
        return pd.rolling_std(df_in, abs(periods), min_periods=abs(periods))

    

pset = gp.PrimitiveSetTyped('MAIN', [pd_df_float, 
                                pd_df_float, 
                                pd_df_float, 
                                pd_df_float,
                                pd_df_float,
                                pd_df_bool,
                                pd_df_bool], 
                                pd_df_bool)

pset.renameArguments(ARG0='Open')
pset.renameArguments(ARG1='High')
pset.renameArguments(ARG2='Low')
pset.renameArguments(ARG3='Close')
pset.renameArguments(ARG4='Volume')
# need to have pd_df_bool terminals for GP to work
pset.renameArguments(ARG5='Ones')
pset.renameArguments(ARG6='Zeros')

pset.addPrimitive(sma, [pd_df_float, int], pd_df_float, name="sma")
pset.addPrimitive(ewma, [pd_df_float, int], pd_df_float, name="ewma")
pset.addPrimitive(hh, [pd_df_float, int], pd_df_float, name="hh")
pset.addPrimitive(ll, [pd_df_float, int], pd_df_float, name="ll")
pset.addPrimitive(pd_std, [pd_df_float, int], pd_df_float, name="pd_std")
pset.addPrimitive(np.log, [pd_df_float], pd_df_float)
pset.addPrimitive(pd_diff, [pd_df_float, int], pd_df_float)
pset.addPrimitive(pd_add,  [pd_df_float, pd_df_float], pd_df_float, name="pd_add")
pset.addPrimitive(pd_subtract,  [pd_df_float, pd_df_float], pd_df_float, name="pd_sub")
pset.addPrimitive(pd_multiply,  [pd_df_float, pd_df_float], pd_df_float, name="pd_mul")
pset.addPrimitive(pd_divide,  [pd_df_float, pd_df_float], pd_df_float, name="pd_div")
pset.addPrimitive(operator.add, [int, int], int, name="add")
pset.addPrimitive(operator.sub, [int, int], int, name="sub")
#pset.addPrimitive(operator.mul, [int, int], int, name="mul")
pset.addPrimitive(protectedDiv, [int, int], int, name="div")

pset.addPrimitive(f_gt, [pd_df_float, float], pd_df_bool )
pset.addPrimitive(f_lt, [pd_df_float, float], pd_df_bool )

pset.addEphemeralConstant("short", lambda: random.randint(2,60), int)
pset.addEphemeralConstant("medium", lambda: random.randint(60,100), int)
pset.addEphemeralConstant("long", lambda: random.randint(100,200), int)
pset.addEphemeralConstant("xtralong", lambda: random.randint(200,20000), int)
pset.addEphemeralConstant("rand100", lambda: random.randint(0,100), int)
pset.addEphemeralConstant("randfloat", lambda: np.random.normal() / 100. , float)

pset.addPrimitive(operator.lt, [pd_df_float, pd_df_float], pd_df_bool)
pset.addPrimitive(operator.gt, [pd_df_float, pd_df_float], pd_df_bool)
pset.addPrimitive(np.bitwise_and, [pd_df_bool, pd_df_bool], pd_df_bool)
pset.addPrimitive(np.bitwise_or, [pd_df_bool, pd_df_bool], pd_df_bool)
pset.addPrimitive(np.bitwise_xor, [pd_df_bool, pd_df_bool], pd_df_bool)
pset.addPrimitive(np.bitwise_not, [pd_df_bool], pd_df_bool)
pset.addPrimitive(operator.add, [float, float], float, name="f_add")
pset.addPrimitive(operator.sub, [float, float], float, name="f_sub")
pset.addPrimitive(protectedDiv, [float, float], float, name="f_div")
pset.addPrimitive(operator.mul, [float, float], float, name="f_mul")

#Better to pass this terminals as arguments (ARG5 and ARG6)
#pset.addTerminal(pd.TimeSeries(data=[1] * len(train), index=train.index, dtype=bool), pd_df_bool, name="ones")
#pset.addTerminal(pd.TimeSeries(data=[0] * len(train), index=train.index, dtype=bool), pd_df_bool, name="zeros")

pset.addTerminal(1.618, float)
pset.addTerminal(0.1618, float)
pset.addTerminal(0.01618, float)
pset.addTerminal(0.001618, float)
pset.addTerminal(-0.001618, float)
pset.addTerminal(-0.01618, float)
pset.addTerminal(-0.1618, float)
pset.addTerminal(-1.618, float)
pset.addTerminal(1, int)


for p in primes:
    pset.addTerminal(p, int)
    
for f in np.arange(0,0.2,0.002):
    pset.addTerminal(f, float)
    pset.addTerminal(-f, float)


creator.create('FitnessMax', base.Fitness, weights=(1.0,))
creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=1, max_=6)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('compile', gp.compile, pset=pset)

    
        

def evalFitness(individual, points):
    func = toolbox.compile(expr=individual)
    s = func(points.Open, points.High, points.Low, points.Close, points.Volume, points.Ones, points.Zeros)
    # transform from bool to int
    s = s*1    
    w = (s * points.Close.diff()) - np.abs(s.diff())*cost
    w.dropna(inplace=True)

#    W_win = w[w>0].sum()
#    W_lose = abs(w[w<0].sum())
#    
#    profit_factor = protectedDiv(W_win, W_lose)        
#    return profit_factor ,  
    
    sharpe = w.mean() / w.std() * math.sqrt(600*255)

    if np.isnan(sharpe) or np.isinf(sharpe):
        sharpe = -99999
    
    return sharpe,      
    

    
toolbox.register('evaluate', evalFitness, points=train)
toolbox.register('select', tools.selTournament, tournsize=3)
toolbox.register('mate', gp.cxOnePoint)
toolbox.register('expr_mut', gp.genFull, min_=0, max_=3)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


def plot(individual):
    nodes, edges, labels = gp.graph(individual)
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    pos = nx.drawing.nx_agraph.graphviz_layout(g, prog="dot")
    nx.draw_networkx_nodes(g, pos)
    nx.draw_networkx_edges(g, pos)
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()

    

if __name__ == '__main__':
    #random.seed(10)
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('min', np.min)
    stats.register('max', np.max)
    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, 160, 160, 0.6, 0.1, 50, stats=stats, halloffame=hof)
    # get the info of best solution
    print("Best solution found...")
    print(hof[0])
    plot(hof[0])
    f=toolbox.compile(hof[0])
    
    # Check training results
    s=f(train.Open, train.High, train.Low, train.Close, train.Volume, train.Ones, train.Zeros)
    s=s*1
    w = (s * train.Close.diff()) - np.abs(s.diff())*cost
    
    W = w.cumsum()

    df_plot = pd.DataFrame(index=train.index)
    df_plot['GP Strategy'] = W
    df_plot['IBEX'] = train.Close
    #Normalize to 1 the start so we can compare plots.
    df_plot['IBEX'] = df_plot['IBEX'] / df_plot['IBEX'][0]
    df_plot.plot()     
    # Check testing results
    s=f(test.Open, test.High, test.Low, test.Close, test.Volume, test.Ones, test.Zeros)
    s=s*1
    w = (s * test.Close.diff()) - np.abs(s.diff())*cost
    
    W = w.cumsum()

    df_plot = pd.DataFrame(index=test.index)
    df_plot['GP Strategy'] = W
    df_plot['IBEX'] = test.Close
    #Normalize to 1 the start so we can compare plots.
    df_plot['IBEX'] = df_plot['IBEX'] / df_plot['IBEX'][0]
    df_plot.plot()     
