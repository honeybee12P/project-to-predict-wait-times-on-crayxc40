from __future__ import division
import math, sys, os
from time import clock, time
from pylab import *
from operator import mul
from scipy.integrate import quad
from pylab import *



def cost_interval(interval):
    area = integral(interval[0], interval[1])
    return area

#[
#[[0.0, 0.023255813953488372], [0.046511627906976744, 0.11627906976744186], [0.18604651162790697, 0.3488372093023256]], 
#[[0.0, 0.32558139534883723], [0.46511627906976744, 0.5116279069767442], [0.6976744186046512, 0.9302325581395349], [0.9534883720930233, 1.0]]
#] 
#[((2, 3), 3), ((1, 2), 2), ((3, 4), 2), ((0, 1), 1), ((4, 5), 1)]


def coverage_area(rangeSet, threshold):
    #print "coverage_area:",rangeSet
    coverageArea = 0.0
    a = []
    for range1 in rangeSet:
      
      for nr in range1:
         a.append(nr)
         
    for s in a:  
       coverageArea += min(threshold, s[1])*(s[0][1]-s[0][0])
    return coverageArea

def cost_rangeSet(rangeSet, threshold):
    # Input is of the form [((2, 3), 3), ((1, 2), 2), ((3, 4), 2), ((0, 1), 1), ((4, 5), 1)]
    cost = 0.0
    for item in rangeSet:
        #print "item:",item[0][1],item[0][0]
        cost += min(item[0][1], threshold)*cost_interval(item[0][0])
    # print cost
    return cost

def cost_allRangeSets(setRangeSet, threshold):
    threshold = 1
    costList = []
    #print "set:",setRangeSet
    for rangeSet in setRangeSet:
        #print "first:",rangeSet
        currentCost = cost_rangeSet(rangeSet, threshold)
        # find coverage length of this range. Max(range) - Min(range)
        rangeMin = float('inf')
        rangeMax = 0
        for rangeItem in rangeSet:
            if rangeMin >= rangeItem[0][0][0]:
                rangeMin = rangeItem[0][0][0]
            if rangeMax <= rangeItem[0][0][1]:
                rangeMax = rangeItem[0][0][1]
        currentCost /= coverage_area(rangeSet, threshold)
        costList.append(currentCost)
    # print "costlist:",costList
    return costList


def normalise(rangeSet):
    #print "range--------:",rangeSet
    max_value = 0; min_value=float('inf');
    for r in rangeSet:
        for r2 in r:
                # print r2
                #print r2[0][0][0]
		if r2[0][0][0] == r2[0][0][1]:
			r2[0][0][1] += 900
    for r in rangeSet:
        for r1 in r:
            if min_value >= r1[0][0][0]:
                min_value = r1[0][0][0]
            if max_value <= r1[0][0][1]:
                max_value = r1[0][0][1]

    diff = max_value-min_value

    for r in rangeSet:
        for r1 in r:
            r1[0][0][0] = (r1[0][0][0]-min_value)/diff
            r1[0][0][1] = (r1[0][0][1]-min_value)/diff
    #print rangeSet
    return rangeSet

def function1(x):
    return (1-x)

def integral(val1, val2):
    result, err = quad(function1, val1, val2)
    return result

