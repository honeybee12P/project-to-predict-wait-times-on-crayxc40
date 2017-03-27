from __future__ import division
from collections import Counter
import math, sys, os
from time import clock, time
from pylab import *
from operator import mul
import gain as gain
EnabledMetaSchedulling = True

def final_request_size(ranges):
		req_size_list = [] 
		max_gain = max(ranges.values())
		for i in ranges.keys():
			if max_gain == ranges[i]:
				req_size_list.append(i)
		final_req_size = min(req_size_list)
                
		return final_req_size

def gain_calculation(processor_list,rangeSet):
            threshold = 3
            #print "input_to_gain:",rangeSet
            if( len(rangeSet) > 1):
             finalrangeSet1=[]
             finalrangeSet1.append(rangeSet)
             #print "input_to_gain:",finalrangeSet1
	     final_rangeSet = gain.normalise(finalrangeSet1) 
            # print "after_normalisation_range_set:",final_rangeSet                #Calling of Normalize function in Gain Module.
	     cost_list = gain.cost_allRangeSets(final_rangeSet,threshold)         #calculation of gain for the whole Job containing
             
	     return cost_list
            else:
             return 0.5
       
def Pick_value(processor_list,waittime_list,estimated_runtims):
         a = estimated_runtims
         b = waittime_list
         #print "a:",a
         #print "b:",b
         
         ga_list = []
         #print "inputlist:",processor_list,waittime_list,estimated_runtims,len(estimated_runtims)
         newlist = [];newpro = [];new_finalarray = []
         '''
         if ( len(a) != 1):
          if( len(a) != 2):
           for k in range(len(a)):
             for i in a[k]:
              newlist.append(i[0])
              newpro.append(i[1])
              new_array = ([[[i+b[k],j+b[k]],g] for (i,j),g in zip(newlist,newpro)])
              newlist[:] = []
              newpro[:] = []
              new_finalarray.append(new_array)
           #print "output_to_gain:",processor_list,new_finalarray
          
           ga_list,proc_list= gain_calculation(processor_list,new_finalarray)
           new_gain = dict(zip(proc_list, ga_list))
           final_proc_size = final_request_size(new_gain)  
           return final_proc_size
          
         
          elif(len(a[0][0]) > 1):
           a = estimated_runtims
           b = waittime_list
           newlist.append(a[0][0][0])
           newpro.append(a[0][0][1])
           new_array = ([[[[i+b[0],j+b[0]],1]] for (i,j) in newlist])
           #print "new_array:",new_array 
           ga_list,proc_list= gain_calculation(processor_list,new_array)
           new_gain = dict(zip(proc_list, ga_list))
           final_proc_size = final_request_size(new_gain)  
           return final_proc_size 
         else:
           return processor_list[0]
         '''
         if ( len(a) > 0):
          
           for k in range(len(a)):
             for i in a[k]:
              newlist.append(i[0])
              newpro.append(i[1])
              new_array = list([[i+b[k],j+b[k]],g] for (i,j),g in zip(newlist,newpro))
              newlist[:] = []
              newpro[:] = []
              new_finalarray.append(new_array)
         
             
             ga_val = gain_calculation(processor_list,new_finalarray)
             new_finalarray[:] = []
             ga_list.append(ga_val)
           #print "returned_costvalue:",ga_list
           new_gain = dict(zip(processor_list, ga_list))
           #print "cost_list,processor_list:",processor_list, ga_list,len(processor_list),len(ga_list)
           #print "gain_value_input:",new_gain
           final_proc_size = final_request_size(new_gain)  
           return final_proc_size
         else:
           return processor_list[0]
