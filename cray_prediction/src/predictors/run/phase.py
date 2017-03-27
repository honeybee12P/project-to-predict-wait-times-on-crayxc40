from __future__ import division
import math, sys, os
from time import clock, time
from pylab import *
from operator import mul


def phase_fun(job_num,min_range,max_range,actual_data):
    count = 0;nmin = [];val = [];b = 0;array_of_min = [];nmax = [];array_of_max = [];new_result_min = [];new_result_max = [];apparr= []
    val1_arr = [];val2_arr = [];new_array =[];ranges = [];endpoints = []; a = [];result = [];final_results = [];output1 = [];output = []
    arr = [];pro = [];narr = [];val_pt = [];narr1 = [];newax = [];nout = []
    for i in range(len(min_range)):
        a.append([min_range[i],max_range[i]])
    ranges = sorted(a, key=lambda x: x[0])
                        
    for r in ranges:
     start, end = r[0], r[1]
     endpoints.append((start, -1))
     endpoints.append((end, 1))

    endpoints.sort()
    cumsum = 0
    result_endpoints = []

    for point, value in endpoints:

      if cumsum == 0 and value == -1:
        result_endpoints.append(point)

      cumsum = cumsum + value

      if cumsum == 0 and value == 1:
        result_endpoints.append(point)
   
    result = zip(result_endpoints[0::2], result_endpoints[1::2])
    
    for i in range(len(result)):
        final_results.append([(result[i][0],result[i][1]),0])
    final_results_copy = final_results
    final_dict = dict(final_results)
   
    for  hruntime in actual_data:
        for r1 in final_results_copy:
            counter_range = final_dict[(r1[0][0],r1[0][1])]
            if r1[0][0] <= hruntime and r1[0][1] > hruntime:
                counter_range += 1
                final_dict[(r1[0][0],r1[0][1])] = counter_range
    final_results = [];
    for i,v in final_dict.items():
       if(v == 0):
        v = 1
        final_results.append((list(i),v))
       elif(v == 1):
        v = 2
        final_results.append((list(i),v))
       else:
        final_results.append((list(i),v))
    sorted_results = sorted(final_results, key=lambda x: x[1], reverse=True)
    for r1 in sorted_results:                        #if you do not want the pripority to display
        output.append(r1[0])
    if len(sorted_results) > 13:                        #if you do not want the pripority to display
        
    	for a1 in range(13):
		output1.append(sorted_results[a1])
	        nout.append(output[a1])
        for i,j in output1:
          arr.append(i)
          pro.append(j*0.1)

        for i in range(len(arr)):
            narr.append(arr[i][0])
            narr1.append(arr[i][1])
        for k in range(len(narr)):
            newax.append(narr[k]+narr1[k])

        for j in range(len(pro)):
          val_pt.append(newax[j]*pro[j])
    	return output1,sum(val_pt),nout
    else:
        for i,j in sorted_results:
          arr.append(i)
          pro.append(j*0.1)

        for i in range(len(arr)):
            narr.append(arr[i][0])
            narr1.append(arr[i][1])
        for k in range(len(narr)):
            newax.append(narr[k]+narr1[k])


        for j in range(len(pro)):
          val_pt.append(newax[j]*pro[j])
    
    	return sorted_results,sum(val_pt),output

    
       

if __name__ == '__main__':
    vou,tr,naout = phase_fun(job_num,namin,namax,hrun)
