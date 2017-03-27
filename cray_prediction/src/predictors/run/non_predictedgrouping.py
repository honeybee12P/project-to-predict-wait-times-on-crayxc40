import math, sys, os
from numpy import array
def non_predictedfunction_shift(target_job,target_load,predicted_load,predicted_run,nnx):
	        
            
                comin1 = [];comin2 = [];comax1 = [];comax2 = [];rmin2 = [];rmax2 = [];new_fun = []
                min_f = [];max_f = [];fun_min = [];fun_max = [];diffmin = [];diffmax = [];new_run = []
                new_load = [];newapprun = [];newappload = [];a1 = [];b1 = []
                r1 = predicted_run
                hl = predicted_load
                tload = target_load
                f1 = open("./predictors/run/standard_function.txt","r")
                lis = f1.readlines()
                for li in lis:
                 injob = li.split()
                 comin1.append(float(injob[0]))
                 comin2.append(float(injob[1]))
              
                for i in range(len(comin1)):
                   rmin2.append((comin1[i]*(hl))+comin2[i])
                   
                for j in range(len(rmin2)):
                   diffmin.append((r1 - rmin2[j]))
                 
                for k in range(len(diffmin)):
                  fun_min.append((comin1[k]*(tload))+comin2[k]+ diffmin[k])
                for l in range(len(fun_min)):
                  if( fun_min[l] > 0):
                    new_fun.append(fun_min[l])
                if ( len(new_fun) != 0):
                  a = sorted(new_fun)
                  b = a[:len(a)/2]
                  c = a[len(a)/2:]
                 
                 
                  bb = list(reversed(b))
                    
                  if( len(bb) != 0): 
                    for i in range(len(bb)):
                     v = (abs(bb[i]-c[i])/c[i])
                     if( v <= nnx ):
                      a1.append(bb[i])
                      b1.append(c[i])
                    if( len(a1) != 0):
                      return a1[0],b1[0],r1
                    else:
                      return b[0],c[0],r1
                  else:
                      return 0,0,0
                else:
                      return 0,0,0
