#!/usr/bin/env python2.4

from base.prototype import JobSubmissionEvent, JobTerminationEvent, JobPredictionIsOverEvent  
from base.prototype import ValidatingMachine
from base.event_queue import EventQueue
from common import CpuSnapshot, list_print

from easy_plus_plus_scheduler import EasyPlusPlusScheduler
from shrinking_easy_scheduler import ShrinkingEasyScheduler

from simulator import Simulator  
import time
import Maingain as Mainval
import est_runtime_assump as assumpt
EnabledWaitPred = True
EnabledRunPred = True

if EnabledRunPred:
	from predictors.run import RuntimePrediction
	RunPredictor = RuntimePrediction.Batch_System	 
        

if EnabledWaitPred:
	from predictors.wait import QueueWaitingTime
	WaitPredictor = QueueWaitingTime.Batch_System

import math
import sys
import random


class MetaScheduler(object):
    """
    This class is the common interface between the User, Predictor and Site Scheduler
    Tasks:
     - Accept job inputs from user
     - Obtain predictions
     - Submit jobs to scheduler
     - Provide feedback (if necessary) to user and predictor
    """

    def __init__(self, jobs, num_processors, scheduler):
        
        self.num_processors = num_processors
        self.jobs = jobs
	print "jobs",jobs
        #print len(self.jobs)
        self.terminated_jobs=[]
        self.scheduler = scheduler
        self.time_of_last_job_submission = 0
        self.event_queue = EventQueue()
        #with open('/home/sharath/Desktop/pyss-read-only/src/5K_sample') as f:
         #  a = sum(1 for _ in f)
        #print "value of history:",a
        self.historySetSize = 100
        self.currentSubmitCount = 0

	if EnabledWaitPred:
            self.waitTimePredictor = WaitPredictor()
	else:
            self.waitTimePredictor = None

        if EnabledRunPred:
            self.runTimePredictor = RunPredictor()
	    
        else:
            self.runTimePredictor = None

        self.machine = ValidatingMachine(num_processors=num_processors, event_queue=self.event_queue, wait_predictor=self.waitTimePredictor, run_predictor = self.runTimePredictor, scheduler = self.scheduler)

        self.event_queue.add_handler(JobSubmissionEvent, self.handle_submission_event)
        self.event_queue.add_handler(JobTerminationEvent, self.handle_termination_event)
        
        if isinstance(scheduler, EasyPlusPlusScheduler) or isinstance(scheduler, ShrinkingEasyScheduler):
            self.event_queue.add_handler(JobPredictionIsOverEvent, self.handle_prediction_event)
        
        countSubmissions = 0    
        for job in self.jobs:
            countSubmissions += 1
            self.event_queue.add_event( JobSubmissionEvent(job.submit_time, job) )     
        #print '** Added', countSubmissions, 'job submission events to event queue **'
        self.queueSimulator = Simulator(jobs, num_processors, scheduler, self.event_queue, self.machine, )

    def modify_job_attributes(self, event, newRequestSize, actual_runtime, actual_ert):
	
	    oldRunTime = event.job.actual_run_time
	    event.job.num_required_processors = newRequestSize
	    event.job.user_estimated_run_time = actual_ert
	    event.job.predicted_run_time = actual_ert
	    if actual_runtime == 0:
		event.job.actual_run_time = actual_ert
	    else:
                event.job.actual_run_time = actual_runtime
           

    def change_job_attributes(self, event, newRequestSize,actual_ert):
            oldRunTime = event.job.actual_run_time
	    event.job.num_required_processors = newRequestSize
	    #event.job.user_estimated_run_time = actual_ert
	    #event.job.predicted_run_time = actual_ert
	    

    def decision_metrics(self, event,queuedJobs,runningJobIDs,allrunningjobevents,jobtarrun):
        
        #print "event:",event
	
       
        #queuedJobIDs = [j.id for j in self.event_queue.QueuedJobs]
        #runningJobIDs = [j.id for j in self.event_queue.RunningJobs]
        #queuedJobIDs.append(event.job.id)
       
        originalRequestSize = event.job.num_required_processors
        waitPredictions = {}
        responsePredictions = {}
	waittime_list = []
        run_list = []
        wait_list = []
        submittime = []
        #terminateval = []
	ifcounter = 0
        if EnabledRunPred:
            run_list.append(jobtarrun)
            submittime.append(event.job.submit_time)
            waitPredictions = self.waitTimePredictor.get_prediction(event.job)
            wait_list.append(waitPredictions)
            pred_run,runid = self.runTimePredictor.get_predictiondelay(event.job)
            
            
            if(len(runid) > 1):
             
             for i in runid:
              ifcounter += 1
              
              for obj in allrunningjobevents:
             	 if obj.id == i in runningJobIDs:
                       terminateval = obj
              
              runningJobIDs.remove(i)
            
              relevantRunningObjs = []
              
	      for obj in allrunningjobevents:
             	 if obj.id in runningJobIDs:
	             relevantRunningObjs.append(obj)
             
              self.runTimePredictor.notify_job_termination_event(terminateval)
              
               
              if EnabledWaitPred:
                    
                    self.waitTimePredictor.notify_arrival_event(event.job, queuedJobs, relevantRunningObjs) 
                    self.runTimePredictor.notify_arrival_event(event.job, queuedJobs, relevantRunningObjs)
                    #if self.currentSubmitCount > self.historySetSize:
                    #print "afterifcondition:"
                    #print "counter:",ifcounter
                    waitPredictions = self.waitTimePredictor.get_prediction(event.job)#to obtain waittime prediction for all running jobs
                    wait_list.append(waitPredictions)
                    #print "wait_prediction:",waitPredictions
                    RunPredictions  = self.runTimePredictor.get_actprediction(event.job)#to obtain runtime prediction for all running jobs
                    #print "run_prediction:",RunPredictions
                    run_list.append(RunPredictions)
                    submittime.append(event.job.submit_time+RunPredictions)
                 
                    runningJobIDs.append(i)
                
             delay = [];decision_list = []
             if( len(wait_list) != [] and len(wait_list) != []):
              l = submittime[0]
              for ru in range(len(run_list)):
                delay.append(abs(l-submittime[ru]))
              
              for k in range(len(wait_list)):
                decision_list.append(wait_list[k]+run_list[k]+delay[k])
              #print "eventjob:",event.job.id
              cjob = event.job.id
              runid.insert(0,cjob)
              inmade = decision_list.index(min(decision_list))
              '''
              print "delay_list:",delay
              print "decision_list:",decision_list,inmade,runid
              print "current_job_id:",cjob
              print "final_jobidsubmission_id:",runid[inmade]
              '''
              return runid,cjob,runid[inmade],inmade,delay
             else:
              empty = []
              return empty,event.job.id,event.job.id,1,empty
            else:
              empty = []
              return empty,event.job.id,event.job.id,1,empty
#study from here

    def handle_submission_event(self, event):
	#print "event",event
        assert isinstance(event, JobSubmissionEvent)
	#print "eventnew",event 
        self.currentSubmitCount += 1
        queuedJobs = self.event_queue.QueuedJobs[:]
        queuedJobs.append(event.job)
	#print "event.job.num_required_processors",event.job.num_required_processors
        originalRequestSize = event.job.num_required_processors
	#print "event.job.actual",event.job.actual_run_time
        waitPredictions = {}
        responsePredictions = {}
	waittime_list = []
	

        if EnabledRunPred:
            #print "parent_jobs:",event.job
            self.runTimePredictor.notify_arrival_event(event.job, queuedJobs, self.event_queue.RunningJobs) 
            self.waitTimePredictor.notify_arrival_event(event.job, queuedJobs, self.event_queue.RunningJobs)          
        self.queueSimulator.handle_submission_event(event)
        
        
    def handle_termination_event(self, event):
        assert isinstance(event, JobTerminationEvent)
        #print 'Term event:', event
        #self.scheduler.cpu_snapshot.printCpuSlices()
        self.queueSimulator.handle_termination_event(event)
        #queuedJobIDs = [j.id for j in self.event_queue.QueuedJobs]
        #runningJobIDs = [j.id for j in self.event_queue.RunningJobs]

        if EnabledWaitPred:
            self.waitTimePredictor.notify_job_termination_event(event.job, self.event_queue.QueuedJobs, self.event_queue.RunningJobs)
	if EnabledRunPred:
            self.runTimePredictor.notify_job_termination_event(event.job)

    def handle_prediction_event(self, event):
        assert isinstance(event, JobPredictionIsOverEvent)
        self.queueSimulator.handle_prediction_event(event)
      
    def run(self):
        while not self.event_queue.is_empty:
            print_simulator_stats(self.queueSimulator)
            self.event_queue.advance()

def runMetaScheduler(num_processors, jobs, scheduler):
    print "In schedulers one 1"
    metaScheduler = MetaScheduler(jobs, num_processors, scheduler)
    metaScheduler.run()
    print_simulator_stats(metaScheduler.queueSimulator)
    return metaScheduler

def print_simulator_stats(simulator):
    simulator.scheduler.cpu_snapshot._restore_old_slices()
    # simulator.scheduler.cpu_snapshot.printCpuSlices()
    #print_statistics(simulator.terminated_jobs, simulator.time_of_last_job_submission)

# increasing order 
by_finish_time_sort_key   = (
    lambda job : job.finish_time
)

# decreasing order   
#sort by: bounded slow down == max(1, (float(wait_time + run_time)/ max(run_time, 10))) 
by_bounded_slow_down_sort_key = (
    lambda job : -max(1, (float(job.start_to_run_at_time - job.submit_time + job.actual_run_time)/max(job.actual_run_time, 10)))
)

    
def print_statistics(jobs, time_of_last_job_submission):
    assert jobs is not None, "Input file is probably empty."
    maxwt = [];max_runt = []
    nwait_sum = nrun_sum = 0
    nwait_sum1 = nrun_sum1 = 0
    nwait_sum2 = nrun_sum2 = 0
    nwait_sum3 = nrun_sum3 = 0
    nwait_sum4 = nrun_sum4 = 0
    nwait_sum5 = nrun_sum5 = 0
    nwait_sum6 = nrun_sum6 = 0
    sum_waits     = 0
    sum_run_times = 0
    sum_slowdowns           = 0.0
    sum_bounded_slowdowns   = 0.0
    sum_estimated_slowdowns = 0.0
    sum_tail_slowdowns      = 0.0
    target_start_id = 10000
    coun = coun1 = coun2 = coun3 = coun4 = coun5 = coun6 = 0
    counter = tmp_counter = tail_counter = 0
    
    size = len(jobs)
    precent_of_size = int(size / 100)
    
    for job in sorted(jobs, key=by_finish_time_sort_key):
        tmp_counter += 1
        if job.id < target_start_id:
            continue  
            

        if job.user_estimated_run_time == 1 and job.num_required_processors == 1: # ignore tiny jobs for the statistics
            size -= 1
            precent_of_size = int(size / 100)
            continue
        
        if size >= 100 and tmp_counter <= precent_of_size:
            continue
        
        if job.finish_time > time_of_last_job_submission:
            break 
        
        counter += 1
        
        wait_time = float(job.start_to_run_at_time - job.submit_time)
        run_time  = float(job.actual_run_time)
        estimated_run_time = float(job.user_estimated_run_time)

        maxwt.append(wait_time)
        max_runt.append(run_time)
        if( len(maxwt) != 0):
         sum_waits += wait_time
         sum_run_times += run_time
         sum_slowdowns += float(wait_time + run_time) / run_time
         sum_bounded_slowdowns   += max(1, (float(wait_time + run_time)/ max(run_time, 10))) 
         sum_estimated_slowdowns += float(wait_time + run_time) / estimated_run_time
         if(abs(job.start_to_run_at_time-job.finish_time) > 0 and abs(job.start_to_run_at_time-job.finish_time) <= 3600):
            nwait_sum += wait_time
            nrun_sum += run_time
            coun += 1
         elif(abs(job.start_to_run_at_time-job.finish_time) > 3600 and abs(job.start_to_run_at_time-job.finish_time) <= 10800):
            nwait_sum1 += wait_time
            nrun_sum1 += run_time
            coun1 += 1
         elif(abs(job.start_to_run_at_time-job.finish_time) > 10800 and abs(job.start_to_run_at_time-job.finish_time) <= 21600):
            nwait_sum2 += wait_time
            nrun_sum2 += run_time
            coun2 += 1
         elif(abs(job.start_to_run_at_time-job.finish_time) > 21600 and abs(job.start_to_run_at_time-job.finish_time) <= 32400):
            nwait_sum3 += wait_time
            nrun_sum3 += run_time
            coun3 += 1
         elif(abs(job.start_to_run_at_time-job.finish_time) > 32400 and abs(job.start_to_run_at_time-job.finish_time) <= 43200):
            nwait_sum4 += wait_time
            nrun_sum4 += run_time
            coun4 += 1
         elif(abs(job.start_to_run_at_time-job.finish_time) > 43200 and abs(job.start_to_run_at_time-job.finish_time) <= 86400):
            nwait_sum5 += wait_time
            nrun_sum5 += run_time
            coun5 += 1
         elif(abs(job.start_to_run_at_time-job.finish_time) > 86400):
            nwait_sum6 += wait_time
            nrun_sum6 += run_time
            coun6 += 1
        if max(1, (float(wait_time + run_time)/ max(run_time, 10))) >= 3:
            tail_counter += 1
            sum_tail_slowdowns += max(1, (float(wait_time + run_time)/ max(run_time, 10)))
            
    sum_percentile_tail_slowdowns = 0.0
    wait_time = float(job.start_to_run_at_time - job.submit_time)
    percentile_counter = counter
    
    for job in sorted(jobs, key=by_bounded_slow_down_sort_key):
        run_time  = float(job.actual_run_time)
        sum_percentile_tail_slowdowns += float(wait_time + run_time) / run_time
        percentile_counter -= 1 # decreamenting the counter 
        if percentile_counter < (0.9 * counter):
            break
        
        
        
    print
    if( len(maxwt) != 0):
     print "STATISTICS: "
    
     print "Wait (Tw) [minutes]: ", float(sum_waits) / (60 * max(counter, 1))

     print "Response time (Tw+Tr) [minutes]: ", float(sum_waits + sum_run_times) / (60 * max(counter, 1))
    
     print "min_qwt,max_qwt:",min(maxwt),max(maxwt)
     print "min_responsetime,max_responsetime:",min(max_runt),max(max_runt)

     print "Slowdown (Tw+Tr) / Tr: ", sum_slowdowns / max(counter, 1)

     print "Bounded slowdown max(1, (Tw+Tr) / max(10, Tr): ", sum_bounded_slowdowns / max(counter, 1)
    
     print "Estimated slowdown (Tw+Tr) / Te: ", sum_estimated_slowdowns / max(counter, 1)
 
     print "Tail slowdown (if bounded_sld >= 3): ", sum_tail_slowdowns / max(tail_counter, 1)
     print "   Number of jobs in the tail: ", tail_counter

     print "Tail Percentile (the top 10% sld): ", sum_percentile_tail_slowdowns / max(counter - percentile_counter + 1, 1)    
    
     print "Total Number of jobs: ", size
    
     print "Number of jobs used to calculate statistics: ", counter


     print "ranges <1(wait,run,counter):",nwait_sum,nrun_sum,coun
     print "ranges 1-3(wait,run,counter):",nwait_sum1,nrun_sum1,coun1
     print "ranges 3-6(wait,run,counter):",nwait_sum2,nrun_sum2,coun2
     print "ranges 6-9(wait,run,counter):",nwait_sum3,nrun_sum3,coun3
     print "ranges 9-12(wait,run,counter):",nwait_sum4,nrun_sum4,coun4
     print "ranges 12-1day(wait,run,counter):",nwait_sum5,nrun_sum5,coun5
     print "ranges >1day(wait,run,counter):",nwait_sum6,nrun_sum6,coun6
     print "--- seconds ---:",time.time() - start_time
     print
     
