from __future__ import division
from collections import Counter
import math, sys, os
from time import clock, time
from pylab import *
from operator import mul
import random

def runtime_random(min_runtime, max_runtime, history_runtime_list):
    rangeSet = {}; rangeSet_list =[]
    number_of_intervals = 10
    interval_size = (max_runtime - min_runtime)/number_of_intervals
    interval_list =[]; size =0; sum1=0;
    interval_list.append(min_runtime)
    for i in range(number_of_intervals-1):
	size += interval_size
	interval_list.append(size)
    interval_list.append(max_runtime)
    counter =0
    for i in range(number_of_intervals):
	rangeSet[(interval_list[i],interval_list[i+1])] = counter
    
    for r in rangeSet.keys():
	for runtime in history_runtime_list:
	    if r[0] <= runtime and r[1] > runtime:
		rangeSet[r] += 1

    '''
    For calculating the probability of each range

    for r in rangeSet.values():
	sum1 += r
    for u in rangeSet.keys():
	rangeSet[u] = float(rangeSet[u])/float(sum1)
    '''
    for i,v in rangeSet.items():
	rangeSet_list.append([list(i),v])

    rangeSet_list = sorted(rangeSet_list, key=lambda x: x[1], reverse=True)

    return random.uniform(rangeSet_list[0][0][0], rangeSet_list[0][0][1])



    
