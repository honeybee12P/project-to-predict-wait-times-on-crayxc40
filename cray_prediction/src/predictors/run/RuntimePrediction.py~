from __future__ import division
from collections import Counter
import math, sys, os
from time import clock, time
from pylab import *
from operator import mul
import non_predictedgrouping as nongro
import phase as step
import re
from itertools import islice
EnabledMetaSchedulling = True

class Batch_System:
	
	# Initialize all the variables
	def __init__(self):
		self.__History_Begin = 1 ; self.__History_End = 5000 ; self.__TargetJobs_Begin = self.__History_End + 1 ; self.__Target_Size = 100
                
                self._sucess = 0; self._failure = 0;self.__all = 0
		self.__History_Begin = 1;self.__v = 0;self.__p = 0;self.__c = 0;self.__l = 0;self.__w = 0;self.__s = 0;self.__f = 0
		self.Max_Nodes = 0 ; self.__free_nodes = self.Max_Nodes ; self.__Near_Mid_ShortJobs = 0 ;self.__count = 0
		self.__HISTORY_SIZE = self.__History_End - self.__History_Begin
		self.__good_metric_ibl = 0 ; 
		self.__history_jobs = [] ; self.__target_jobs = [];self.__naivesuccess = 0; self.__naivefailure = 0;
		self.__job_num = {} ; self.__submit_time = {} ; self.__start_time = {} ; self.__run_time = {} ; self.__req_proc = {} ;
                self.__job_num = {} ; self.__submit_time = {} ; self.__start_time = {} ; self.__end_time = {} ;self.__nn_job = {} ;
                self.__run_time = {} ; self.__req_proc = {} ;self.__target_endtime = {} ;self.__wait_time = {} ;self.__group_id = {}
                self.__user_id = {} ;self.__queue_id = {} ;self.__req_time = {};self.__pre_load = {};self.__act_load = {}; self.__target_req_size = 0;
		self.__max_wait_time = 0; self.__min_wait_time= 0;self.__ranges1=[];
 
       

        def __predicted_evidence_window(self,job_id):#To calculate the predicted load
	       
                v1 = self.__submit_time[job_id]    ; v2 = v1 - 604800
                overlap_time = 0;total_CPU = 0;count=0
                value = 0;value1 = 0;value2 = 0;value3 = 0;overall_area= 0
                l1 = self.__submit_time[job_id] ; l2 = self.__target_endtime[job_id]  
		
                for j in self.__history_jobs:
                  if self.__job_num[j] not in self.__waiting_job_id: # If j is not waiting then proceed add an if statement. (Done)

                      if self.__job_num[j] in self.__running_job_id:
			      self.__end_time1 = self.__submit_time[job_id]
		      else:
			      self.__end_time1 = self.__end_time[j]

                      if self.__submit_time[job_id] >= (self.__submit_time[j] + self.__wait_time[j]): 

                       
			      if( self.__start_time[j] < l1 and self.__end_time1 > l2):
                              
                               overlap_time = abs(l1 - l2)
                               total_CPU = self.__req_proc[j]
                               value += (overlap_time * total_CPU)
                               count += 1
                               
                              
			      if( self.__start_time[j] > l1 and self.__end_time1 < l2):
                               overlap_time = abs(self.__start_time[j] - self.__end_time1)
                               total_CPU = self.__req_proc[j]
                               value1 += (overlap_time * total_CPU)        
                               count += 1 
                               
			      if( self.__start_time[j] < l1 and self.__end_time1 > l1 and self.__end_time1 < l2):
                               overlap_time = abs(l1 - self.__end_time1)
                               total_CPU = self.__req_proc[j]
                               value2 += (overlap_time * total_CPU)
                               count += 1
                         
                               
			      if( self.__start_time[j] > l1 and self.__start_time[j] < l2 and self.__end_time1 > l2):
                               overlap_time += abs(self.__start_time[j] - l2)
                               total_CPU += self.__req_proc[j]
                               value3 += (overlap_time * total_CPU)
                               count += 1

                
                
                total_cpu = (800 *abs(l1-l2))
                
                
                if ( total_cpu != 0 ):
                            
                 num = (value+value1+value2+value3)
               
                 tload = (((num) /  ( total_cpu)))
                 round(tload,4)
               
                 return tload
                else:
                
                 return 0
 
        def __actual_evidence_window(self,actual_lo,job_id):#To calculate the actual load
	      
                v1 = self.__submit_time[actual_lo]    ; v2 = v1 - 604800
                overlap_time = 0;total_CPU = 0;count=0
                value = 0;value1 = 0;value2 = 0;value3 = 0;overall_area= 0
                l1 = self.__start_time[actual_lo] ; l2 = self.__end_time[actual_lo]
		
                for j in self.__history_jobs:
                  if self.__job_num[j] not in self.__waiting_job_id:

		     if self.__job_num[j] in self.__running_job_id:
			     self.__end_time1 = self.__submit_time[job_id]
		     else:
			     self.__end_time1 = self.__end_time[j]

		     if self.__submit_time[actual_lo] >= (self.__submit_time[j] + self.__wait_time[j]):
			  # If job is waiting, ignore it
			  # If job is currently running, for now, set its end time to be current time - self.__submit_time[actual_lo]
			     if( self.__start_time[j] < l1 and self.__end_time1 > l2):
                              
			        overlap_time = abs(l1 - l2)
				total_CPU = self.__req_proc[j]
				value += (overlap_time * total_CPU)
				count += 1
                               
                              
			     if( self.__start_time[j] > l1 and self.__end_time1 < l2):
                                overlap_time = abs(self.__start_time[j] - self.__end_time1) # if the job has started at the time of running of the history job and ended before the history job 
                                total_CPU = self.__req_proc[j]
                                value1 += (overlap_time * total_CPU)        
                                count += 1 
                               
			     if( self.__start_time[j] < l1 and self.__end_time1 > l1 and self.__end_time1 < l2):
                                 overlap_time = abs(l1 - self.__end_time1) 
                                 total_CPU = self.__req_proc[j]
                                 value2 += (overlap_time * total_CPU)
                                 count += 1
                         
                               
			     if( self.__start_time[j] > l1 and self.__start_time[j] < l2 and self.__end_time1 > l2):
                                  overlap_time += abs(self.__start_time[j] - l2)
                                  total_CPU += self.__req_proc[j]
                                  value3 += (overlap_time * total_CPU)
                                  count += 1

                
                
                total_cpu = (800 *abs(l1-l2))
                
                
                if ( total_cpu != 0 ):
                            
                  num = (value+value1+value2+value3)
               
                  tload1 = (((num) /  ( total_cpu)))
                  round(tload1,4)
                  return tload1
                else:
                  return 0


        def __predicting_job(self,job_num,tload,hload,hrun,newhruntime,newhisload,predicted_jobnum):
                   amin = [];amax = [];namin = [];namax = [];arun = [];hisrun = [];klist = []
                   hisrunfil = []; new_min = [];new_max = [];sum_val = [];newsum_val = []; a1 = []
                   N = len(newhruntime) * 15
                   nall_val = abs(min(newhruntime) - max(newhruntime))
                   
                   if(len(newhruntime) > 2 and nall_val >= 15):
                    all_val = abs(min(hrun) - max(hrun))
                    ny = all_val/min(hrun)
                    nx = ny/2
                    nnx = nx/N
                    if( nnx < 0.1 ):
                       nnx = 0.1
                    elif(nnx > 0.5):
                       nnx = 0.2
                    else:
                       nnx = nnx
                   
                    for i in range(len(newhisload)):
                      a,b,r1 = nongro.non_predictedfunction_shift(job_num,tload,newhisload[i],newhruntime[i],nnx)#Funtion shift
                      if( a != 0  and b != 0 and r1 != 0):
                       amin.append(a)
                       amax.append(b)
                       hisrun.append(r1)
                           
                    if ( len(amin) != 0):
                     for k in range(len(amin)):
                      if( (abs(amin[k]-amax[k])/amax[k]) <= nnx):
                            namin.append(amin[k])
                            namax.append(amax[k])
                            hisrunfil.append(hisrun[k])
                     if( len(namin) != 0):
                            range_set,poinval,naoutpro= step.phase_fun(job_num,namin,namax,hisrunfil)
                            
                            if EnabledMetaSchedulling:

                               
                               return self.__req_proc[job_num],range_set,self.__req_time[job_num],poinval,naoutpro
                     else:
                      self.__p += 1
                     
                      kmin = min(newhruntime)
                      kmax = max(newhruntime)
                      newrange_set = [[[kmin,kmax],1]]
                      naoutpro = [[kmin,kmax]]  
                      poinval = kmin
                      if EnabledMetaSchedulling:
                 
                       return self.__req_proc[job_num],newrange_set,self.__req_time[job_num],poinval,naoutpro
                    else:
                     self.__v += 1
                    
                     kmin = min(newhruntime)
                     kmax = max(newhruntime)
                     newrange_set = [[[kmin,kmax],1]] 
                     naoutpro = [[kmin,kmax]]
                     poinval = kmin
                     
                     if EnabledMetaSchedulling:
                      
                      return self.__req_proc[job_num],newrange_set,self.__req_time[job_num],poinval,naoutpro
                   else:
                     self.__p += 1
                     kmin = min(newhruntime)
                     kmax = max(newhruntime)
                     newrange_set = [[[kmin,kmax],1]] 
                     naoutpro = [[kmin,kmax]]
                     poinval = kmin
                     if EnabledMetaSchedulling:
                     
                      return self.__req_proc[job_num],newrange_set,self.__req_time[job_num],poinval,naoutpro

        def __fun_disjoint_cond(self,job_num,tload,hload,hrun,predicted_jobnum):#nearness of history
                    newhrun = [];newrangeload = [];newhrun1 = [];newrangeload1 = [];prevminval = [];prevmaxval = []
                 
                    
                    i = 10
                    b = predicted_jobnum[:i]
                    a = hrun[:i]
                    l = hload[:i]
                    iamin = min(a)
                    iamax = max(a)
                    x = len(hrun)-1
                    
                    
                    k = i
                    hamin = min(hrun)
                    hamax = max(hrun)
                 
                    if ( x >= 2 and hamin != hamax):
                    
                     val1 = abs(iamin-iamax)
                     val2 = abs(hamin-hamax)
                  
                     c = val1/val2
                   
                     while( (c <= 0.0 or c <= 0.99) and i <= len(hrun)-10 and x >= 2):
                      k += 1
                      a = hrun[:k]
                      l = hload[:k]
                      iamin = min(a)
                      iamax = max(a)
                      x = len(hrun)-1
                      
                     
                      history_range = hrun[k:x]
                     
                      history_load = hload[k:x]
                      hamin = min(history_range)
                      hamax = max(history_range)
                      
                    
                      val1 = abs(iamin-iamax)
                      val2 = abs(hamin-hamax)
                      if( hamin != hamax ):
                        c = val1/val2
                      else:
                        c = 1.0   
                                 
                     return a,l
                    else:
                     return a,l

                    
        def __range_gen_fun(self,job_num,tload,hload,hrun,predicted_jobnum):
                    

                   nnruncond1 = [];nnloadcond1 = [];nnruncond2 = [];nnloadcond2 = [];
                   his_rang_min = min(hrun)
                   his_range_max = max(hrun)
                   i = 10
                   b = predicted_jobnum[:i]
                   a = hrun[:i]
                   l = hload[:i]
                   if( len(hrun) > 11):
                    newhruntime,newhisload = self.__fun_disjoint_cond(job_num,tload,hload,hrun,predicted_jobnum)
                    
                    
                   
                    if ( len(newhruntime) != 10):
                      
                      pro,runj,esti,poinval,naoutpro= self.__predicting_job(job_num,tload,hload,hrun,newhruntime,newhisload,predicted_jobnum) 
                     
                      return pro,runj,esti,poinval,naoutpro
                    else:
                  
                        
                      pro,runj,esti,poinval,naoutpro = self.__predicting_job(job_num,tload,hload,hrun,newhruntime,newhisload,predicted_jobnum) 
                      
                      return pro,runj,esti,poinval,naoutpro
                   else:
                      pro,runj,esti,poinval,naoutpro = self.__predicting_job(job_num,tload,hload,hrun,hrun,hload,predicted_jobnum)
                 
                      return pro,runj,esti,poinval,naoutpro
                     
                                      
        def __get_runt_strategy_based(self, job_id,predicted_tl):
          
	       predicted_run = [] ; count = 0;c = 0
	       prediction_list = [] ; cluster_submit = [] ;cluster_run = []; predicted_sub = [] ; cluster_jobnum = [] ;
               predicted_jobnum = [] ;cluster_load = []; predicted_load = [] ; final_run = []; final_sub = [] ;cluster_wait = [];
               final_jobnum = []; a = [];newpredicted_run = [] ;newprediction_list = [] ; predicted_wait = []
               newpredicted_load = [];newpredicted_jobnum = [];newpredicted_sub = [];predicted_jnnu = []
	       wait_times = []; ert =[];               
	       for j in self.__history_jobs:
		       #TODO: add an if statement with condition of checking waiting jobs and runnning jobs.
		    
                   if self.__job_num[j] not in self.__waiting_job_id and self.__job_num[j] not in self.__running_job_id:
                           
			   if self.__submit_time[job_id] >= (self.__submit_time[j] + self.__wait_time[j] + self.__run_time[j]): #TODO: the history definition changes so no need of adding the condition for checking of jobs.
                           	
				   if self.__user_id[job_id] == self.__user_id[j] and self.__req_proc[job_id] == self.__req_proc[j] and self.__queue_id[job_id] == self.__queue_id[j]:
                            
                                   
					   predicted_run.append(self.__run_time[j])
					   predicted_sub.append(self.__submit_time[j])
					   predicted_jobnum.append(self.__job_num[j])
					   predicted_jnnu.append(self.__nn_job[j])
					   predicted_wait.append(self.__wait_time[j])
					   wait_times.append(self.__wait_time[j])
						
	       

              
               if(len(predicted_run) != 0):
                    for h in range(len(predicted_jobnum)):
                                   actual = self.__actual_evidence_window(predicted_jnnu[h],job_id)
                                   cluster_load.append(actual)
                    pro,runj,esti,poinval,naoutpro = self.__range_gen_fun(self.__job_num[job_id],predicted_tl,
                   cluster_load,predicted_run,predicted_jobnum)
                    
		    self.__with_history.append(self.__req_proc[job_id])
                    
                    return pro,runj,esti,poinval,naoutpro
                    
               else:  
                 self.__c += 1
                 #print "no_history:",self.__c, self.__req_proc[job_id]
		 self.__no_history.append(self.__req_proc[job_id])
		 return self.__req_proc[job_id],[[[100,100],1]],3600,100,[[100,100]]

        def __job_runt_predictor(self, job_id,predicted_tl):		
		
	 pro,runj,esti,poinval,naoutpro = self.__get_runt_strategy_based(job_id,predicted_tl)
        
         return pro,runj,esti,poinval,naoutpro
        
        def __job_runningalong(self, job_id):
           run_jobsid = []
           for j in self.__history_jobs:
		       #TODO: add an if statement with condition of checking waiting jobs and runnning jobs.
		    
                   if self.__job_num[j] not in self.__waiting_job_id and self.__job_num[j] in self.__running_job_id:
                           run_jobsid.append(self.__job_num[j])
                           #print "running_jobs at:",self.__job_num[j],self.__submit_time[j]+self.__wait_time[j],self.__submit_time[job_id]
           
           rrjo = list(set(run_jobsid))
           
           return rrjo
        def getrunning_prediction(self, Jobidd):
            # The following three attributes may be modified by the meta scheduler
         
	    self.__no_history = []; self.__with_history =[];new_ranges = [];new_pv = []
	  
	    self.__ranges=[];new_estma = [];new_runcal = []
            predicted = self.__predicted_evidence_window(Jobidd)
            #print "predictedload:",predicted
            start_time = self.__start_time[Jobidd]
	    pro,runj,esti,poinval,naoutpro = self.__job_runt_predictor(Jobidd,predicted)
            #print "new_output:",runj            
            return pro,poinval,esti,Jobidd,start_time

        def get_actprediction(self, inJob):
	    #print "in runtime prediction get_prediction"
            myJob = inJob.id
            self.__no_history = []; self.__with_history =[]
	    self.__ranges=[];new_estma = [];new_runcal = [];
            runningjob_list=[];runningjob_prediction = []
            runjobiid = []
	    self.__target_jobs.insert(0,myJob)
          
            predicted = self.__predicted_evidence_window(myJob)
            pro,runj,esti,poinval,naoutpro = self.__job_runt_predictor(myJob,predicted)
            #print "new_output:",runj            
            return poinval

        def __processor_lists(self, job_id): #my whole function
	       final_processor=[];final_processor1=[];
	       self.__target_req_size = self.__req_proc[job_id]                                                                            
	       final_processor.append(self.__req_proc[job_id]) #my

	       for j in self.__history_jobs: 
                   if self.__job_num[j] not in self.__waiting_job_id and self.__job_num[j] not in self.__running_job_id:                     
			   if self.__submit_time[job_id] >= (self.__submit_time[j] + self.__wait_time[j] + self.__run_time[j]):                           	
				   if self.__user_id[job_id] == self.__user_id[j]:

					   final_processor.append(self.__req_proc[j])                                   
                                   
	       final_processor1=list(set(final_processor))
	       
               return final_processor1
        
        
	'''
        Arrival event notification comes from the meta scheduler
        '''
        def notify_arrival_event(self, inJob, waiting_jobs, running_jobs):
            # use the waiting and running(not actually needed) provided from args
            #fileout=open("/home/sharath/Desktop/pyss-read-only/src/filedata","a")
	    
            self.__waiting_job_id = [job.id for job in waiting_jobs]
            #print >>fileout,self.__waiting_job_id
	    self.__running_job_id = [job.id for job in running_jobs]
	    #print >>fileout,self.__running_job_id
            self.__waiting_job_cpu = [job.num_required_processors*job.user_estimated_run_time for job in waiting_jobs]
            #print >>fileout,self.__waiting_job_cpu
            self.__nn_job[inJob.id] = inJob.id
	    #print >>fileout,self.__waiting_job_cpu
            #print "jobidappended:",self.__nn_job
            self.__job_num[inJob.id] = inJob.id   #inJob.id
	    #print >>fileout,self.__waiting_job_cpu
	    self.__submit_time[inJob.id] = inJob.submit_time
            #print >>fileout,self.__waiting_job_cpu
	    self.__req_proc[inJob.id] = inJob.num_required_processors
            #print >>fileout,self.__waiting_job_cpu
	    self.__req_time[inJob.id] = inJob.user_estimated_run_time
            #print >>fileout,self.__waiting_job_cpu
	    self.__user_id[inJob.id] = inJob.user_id
            #print >>fileout,self.__waiting_job_cpu
	    self.__group_id[inJob.id] = inJob.group_id
            #print >>fileout,self.__waiting_job_cpu
	    self.__queue_id[inJob.id] = inJob.queue_id
	    #print >>fileout,self.__waiting_job_cpu
	    self.__target_endtime[inJob.id] = inJob.submit_time + inJob.user_estimated_run_time
	    self.__history_jobs.insert(0,inJob.id)
	    
	    
        '''
        Expected use case is one call to notify_arrival_event and multiple calls to get_prediction
        '''


        def get_predictiondelay(self, inJob):
	    #print "in runtime prediction get_predictiondelay"
            myJob = inJob.id
	    self.__ranges=[];new_estma = [];new_runcal = [];
            runningjob_list=[];runningjob_prediction = []
            runjobiid = [];startlist = [];difflist = []
            final_predictedlist = []
	    self.__target_jobs.insert(0,myJob)
	    
            run_jobs = self.__job_runningalong(myJob)
            actual_jobsubmit = self.__submit_time[myJob]
            #print "target_job_submittime:",actual_jobsubmit
            for i in range(len(run_jobs)):
               
               pro,runj,esti,id_j,start_j = self.getrunning_prediction(run_jobs[i])
               if(runj != 0):
                 runjobiid.append(id_j)
                 runningjob_prediction.append(runj)
                 startlist.append(start_j)
            #print "history_jobstarttime:",startlist
            for i in range(len(startlist)):
                 difflist.append(actual_jobsubmit-startlist[i])
            #print "diff_list:",difflist
            #print "prediction_list:",runningjob_prediction
            for j in range(len(difflist)):
                  final_predictedlist.append(abs(difflist[i]-runningjob_prediction[i]))
            runjobiid = [x for (y,x) in sorted(zip(final_predictedlist,runjobiid))]
            final_predictedlist = sorted(final_predictedlist)
  
            #print "list_of_prediction_for_running_jobs:",final_predictedlist,runjobiid
            return final_predictedlist,runjobiid
       
        def get_prediction(self, inJob):#first call for job molding
            # The following two attributes may be modified by the meta scheduler
            self.__req_proc[inJob.id] = inJob.num_required_processors 
            self.__req_time[inJob.id] = inJob.user_estimated_run_time
	    self.__no_history = []; self.__with_history =[];new_ranges = [];new_pv = []
            new_wtpro = []
	    myJob = inJob.id
	    self.__ranges=[];new_estma = [];new_poinval = []
	    self.__target_jobs.insert(0,myJob)
	   
            
	    final_proc_lists = self.__processor_lists(myJob)#to obtain the processor list
           
            			
	    predicted = self.__predicted_evidence_window(myJob)#predict the load
	    self.__job_id_rangeSet={}
	    if len(final_proc_lists) != 1:								
		    for r9 in range(len(final_proc_lists)):				
			    self.__req_proc[myJob] = final_proc_lists[r9]
                            
                            pro,runj,esti,poinval,naoutpro = self.__job_runt_predictor(myJob,predicted)#for different processor call runtime predictor
                         
                            if ( pro != 0 and runj != [] and esti != []):
                              new_pv.append(pro)
                              new_ranges.append(runj)
                              new_estma.append(esti)
                              new_poinval.append(poinval)
                              new_wtpro.append(naoutpro)
                    if( new_pv != []):   
                    
                     return new_pv,new_ranges,self.__history_jobs,self.__waiting_job_id, self.__running_job_id,self.__job_num,self.__submit_time,self.__req_proc,self.__user_id,self.__run_time,self.__wait_time,self.__req_time,self.__queue_id,myJob,new_estma,new_poinval,new_wtpro  	
		     
                    else:
                     self.__req_proc[myJob] = final_proc_lists[0]
                     new_pv.append(self.__req_proc[myJob])
                     new_ranges.append([[[self.__req_time[myJob],self.__req_time[myJob]],1]])
                     myjobest = [self.__req_time[myJob]]
                    
		     return new_pv,new_ranges,self.__history_jobs,self.__waiting_job_id, self.__running_job_id,self.__job_num,self.__submit_time,self.__req_proc,self.__user_id,self.__run_time,self.__wait_time,self.__req_time,self.__queue_id,myJob,myjobest,100,[[100,100]]     	
		   
                      	    
	    else:

		    self.__req_proc[myJob] = final_proc_lists[0]
                    new_pv.append(self.__req_proc[myJob])
                    new_ranges.append([[[self.__req_time[myJob],self.__req_time[myJob]],1]])
                    myjobest = [self.__req_time[myJob]]
                    			
		    return new_pv,new_ranges,self.__history_jobs,self.__waiting_job_id, self.__running_job_id,self.__job_num,self.__submit_time,self.__req_proc,self.__user_id,self.__run_time,self.__wait_time,self.__req_time,self.__queue_id,myJob,myjobest,100,[[100,100]]    	
		    
            #self.__history_jobs.pop()

                            

        def notify_job_start_event(self, inJob):
	    #print "inside runtime prediction notify_job_start_event"
            self.__start_time[inJob.id] = inJob.start_to_run_at_time 
            self.__wait_time[inJob.id] = self.__start_time[inJob.id] - self.__submit_time[inJob.id]
	    if( len(self.__history_jobs) > 2000):
               self.__history_jobs.pop()

        def notify_job_termination_event(self, inJob):
 	    #print "inside runtime prediction notify_job_termination"
            self.__end_time[inJob.id] = self.__submit_time[inJob.id] + self.__wait_time[inJob.id] + inJob.actual_run_time
            self.__run_time[inJob.id] = inJob.actual_run_time


def main():

		orig_stdout = sys.stdout
                f = file('out4.txt', 'w')
                sys.stdout = f
	        start_time = time()
		system_state = Batch_System()
                system_state.Max_Nodes = int(240)
		
                system_state.convert_input("anl.txt")
                system_state.log_parser()
		

		end_time = time()
	       
                sys.stdout = orig_stdout
                f.close()

