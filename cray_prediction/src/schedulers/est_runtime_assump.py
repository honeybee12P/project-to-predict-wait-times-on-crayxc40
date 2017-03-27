from __future__ import division
from collections import Counter
import math, sys, os
from time import clock, time
from pylab import *
from operator import mul
import roulette as roulette
EnabledMetaSchedulling = True
class Batch_System:
        def __init__(self):
		
                #self.__waiting_job_id = [] ; self.__waiting_job_ert = [] ; self.__waiting_job_reqsize = []  ; self.__waiting_job_cpu = []
		#self.__running_job_id = [] ;
                self._sucess = 0; self._failure = 0;self.__all = 0
		self.__History_Begin = 1;self.__v = 0;self.__p = 0;self.__c = 0;self.__l = 0;self.__w = 0;self.__s = 0;self.__f = 0
		self.Max_Nodes = 0 ; self.__free_nodes = self.Max_Nodes ; self.__Near_Mid_ShortJobs = 0 ;self.__count = 0
		
		self.__good_metric_ibl = 0 ; 
		self.__history_jobs = [] ; self.__target_jobs = [];self.__naivesuccess = 0; self.__naivefailure = 0;
		self.__job_num = {} ; self.__submit_time = {} ; self.__start_time = {} ; self.__run_time = {} ; self.__req_proc = {} ;
                self.__job_num = {} ; self.__submit_time = {} ; self.__start_time = {} ; self.__end_time = {} ;self.__nn_job = {} ;
                self.__run_time = {} ; self.__req_proc = {} ;self.__target_endtime = {} ;self.__wait_time = {} ;self.__group_id = {}
                self.__user_id = {} ;self.__queue_id = {} ;self.__req_time = {};self.__pre_load = {};self.__act_load = {}; self.__target_req_size = 0;
		self.__final_proc_lists=[]; self.__max_wait_time = 0; self.__min_wait_time= 0;self.__final_proc_lists =[];self.__ranges1=[];	

        def ert_runtime_list_1(self,sent_req_size,histjobs,waiting_jobs,running_jobs,jnu,subtm,prore,usid,rtime,wtime,reqt,qid,my_id):
               
                waiting_job_id = waiting_jobs
                running_job_id = running_jobs
                job_num = jnu
	        submit_time = subtm
                req_proc = prore
                user_id = usid
                run_time = rtime
                req_time = reqt
                wait_time = wtime
		actual_runtime = []; actual_ert = [];
                job_id = my_id
	        myJob = my_id
		req_proc[myJob] = sent_req_size
                queue_id = qid 
		for j in histjobs:
                        
			if job_num[j] not in waiting_job_id and job_num[j] not in running_job_id:
                                
				if submit_time[job_id] >= (submit_time[j] + wait_time[j] + run_time[j]): #TODO: the history definition changes so no need of adding the condition for checking of jobs.
                           	       
					if user_id[job_id] == user_id[j] and req_proc[job_id] == req_proc[j] and queue_id[job_id] == queue_id[j]:
                                               
						actual_runtime.append(run_time[j])
						actual_ert.append(req_time[j])

		if(len(actual_ert) != 0):
			max_runtime = max(actual_runtime)
			min_runtime = min(actual_runtime)
                        actual_runtime_roulette = max_runtime
			#actual_runtime_roulette = roulette.runtime_random(min_runtime, max_runtime, actual_runtime)
                        return actual_runtime_roulette, max(actual_ert)
                        
                else: 
                       
                        return max(run_time),max(req_time)

def ert_runtime_list_generator(sent_req_size,histjobs,waiting_jobs,running_jobs,jnu,subtm,prore,usid,rtime,wtime,reqt,qid,my_id):

		
		system_state = Batch_System()
                actual,ert = system_state.ert_runtime_list_1(sent_req_size,histjobs,waiting_jobs, running_jobs,jnu,subtm,prore,usid,rtime,wtime,reqt,qid,my_id)
                
                return actual,ert
               
