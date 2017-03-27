#!/usr/bin/env python2.4

from base.prototype import JobSubmissionEvent, JobTerminationEvent, JobPredictionIsOverEvent
from base.prototype import ValidatingMachine
from base.event_queue import EventQueue
from common import CpuSnapshot, list_print
from schedulers1.metaScheduler import *
from easy_plus_plus_scheduler import EasyPlusPlusScheduler
from shrinking_easy_scheduler import ShrinkingEasyScheduler

from simulator import Simulator
import time
import Maingain as Mainval
import est_runtime_assump as assumpt
EnabledWaitPred = True
EnabledRunPred = True
#start_time = time.time()
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
        #print "inside init of metascheduler"
	#print "jobs",jobs
        #print len(self.jobs)

        self.terminated_jobs=[]
        self.scheduler = scheduler
        self.time_of_last_job_submission = 0
        self.event_queue = EventQueue()

	with open('/home/kruthika/Desktop/Cray Project/cray_prediction/src/cray_log_june30.txt') as f:
           a = sum(1 for _ in f)
       
        
	#self.historySetSize = a - 20810
	self.historySetSize = a - 28946
        #self.historySetSize = 0

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
	    #print "modify_job_attributes"
	    oldRunTime = event.job.actual_run_time
	    event.job.num_required_processors = newRequestSize
	    event.job.user_estimated_run_time = actual_ert
	    event.job.predicted_run_time = actual_ert
	    if actual_runtime == 0:
		event.job.actual_run_time = actual_ert
	    else:
                event.job.actual_run_time = actual_runtime


    def change_job_attributes(self, event, newRequestSize,actual_ert):
	    #print "change job attributes"
            oldRunTime = event.job.actual_run_time
	    event.job.num_required_processors = newRequestSize
	    #event.job.user_estimated_run_time = actual_ert
	    #event.job.predicted_run_time = actual_ert


    def decision_metrics(self, event,queuedJobs,runningJobIDs,allrunningjobevents,jobtarrun):
        #print "inside decision metrics"
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

              return runid,cjob,runid[inmade],inmade,delay
             else:
              empty = []
              return empty,event.job.id,event.job.id,1,empty
            else:
              empty = []
              return empty,event.job.id,event.job.id,1,empty
#study from here

    def handle_submission_event(self, event):
        assert isinstance(event, JobSubmissionEvent)
        self.currentSubmitCount += 1
        #print self.event_queue.QueuedJobs[:]
        queuedJobs = self.event_queue.QueuedJobs[:]
        #print "queued jobs coming from handle submission event 1",queuedJobs
        queuedJobs.append(event.job)
        #print "queued jobs coming from handle submission event 2",queuedJobs
        originalRequestSize = event.job.num_required_processors
        waitPredictions = {}
        responsePredictions = {}
	waittime_list = []


        if EnabledRunPred:
            self.runTimePredictor.notify_arrival_event(event.job, queuedJobs, self.event_queue.RunningJobs)


            if self.currentSubmitCount > self.historySetSize:     #in true means no waiting required? but why?

                processor_list,estimated_runtims,histjobs,wt_jobs,ru_jobs,jnu,subtm,prore,usid,rtime,wtime,reqt,qid,my_id,myest,mypoinval,myranpro = self.runTimePredictor.get_prediction(event.job)#CALL FOR RUNTIME PREDICTION job molding


            if EnabledWaitPred:
              self.waitTimePredictor.notify_arrival_event(event.job, queuedJobs, self.event_queue.RunningJobs)
              #print queuedJobs
	      #if event.job.id == 57051:
	      		#print event.job.id,"\t",self.event_queue.RunningJobs
              if self.currentSubmitCount > self.historySetSize:
	       """
               if( len(processor_list) > 1):  #processor_list: is the list of required processors for jobs by same user which are in history jobs list. For more explaination refer '__processor_lists' in RuntimePrediction.py

                for val in processor_list:
                    self.change_job_attributes(event,val,myest)


                    waitPredictions[val] = self.waitTimePredictor.get_prediction(event.job)		# Getting the wait time prediction lists for different processor


                    waittime_list.append(waitPredictions[val])


                final_proc = Mainval.Pick_value(processor_list,waittime_list,estimated_runtims) #best processor selection based on gain value
                bestRequestSize = final_proc
                #estimated time calculations
                actual_runtime, actual_ert = assumpt.ert_runtime_list_generator(final_proc,histjobs,wt_jobs,ru_jobs,jnu,subtm,prore,usid,rtime,wtime,reqt,qid,my_id)

                #print "Job no:",event.job.id,
		#print " Actual Wait time:",event.job.actual_wait_time,
                #print "Runtime_rangeset:",myranpro[0],'\n'
                #print " Predicted Wait time:",waittime_list[0],' '
                #print "Runtime_pointval:",mypoinval[0],'\n'
                #print "Job Molding:",'\n'
                #print "requested_processor:",processor_list[0],'\n'
                #print "Processor_Selected:",final_proc,'\n'
                #print "Estimated_runtime:",actual_ert,'\n'
                runningJobIDs = [j.id for j in self.event_queue.RunningJobs]
                allrunningjobevents = self.event_queue.RunningJobs
                #call for delayed submission
                id_list,jobcrid,changid,index_val,delay_list = self.decision_metrics(event,queuedJobs,runningJobIDs,allrunningjobevents,mypoinval[0])


                #if(jobcrid == changid):
                #
                #  print "Delayed Submission:",'\n'
                #  print "Delay:",0,'\n'
                #  print "Submit_time:",event.job.submit_time,'\n'
		#
                #else:
                #
                # print "Delayed Submission:",'\n'
                # print "Delay:",min(delay_list),'\n'
                # sub = event.job.submit_time + min(delay_list)
                # print "Submit_time:",sub,'\n'

		self.modify_job_attributes(event, bestRequestSize, actual_runtime, actual_ert)
                self.waitTimePredictor.notify_arrival_event(event.job, queuedJobs, self.event_queue.RunningJobs)
                self.runTimePredictor.notify_arrival_event(event.job, queuedJobs, self.event_queue.RunningJobs)
	       """
               """
               failure_times = [1383055873, 1403892227, 1403352073, 1363341326, 1364919210, 1361532946, 1360346132, 1360606741, 1365394458, 1362193435, 1385859100, 1377049629, 1356359711, 1380556322, 1394757723, 1379864658, 1394765523, 1362449841, 1365344808, 1377053229, 1392908335, 1362835509, 1418507830, 1359058487, 1359867448, 1373525564, 1382614077, 1412846142, 1360951607, 1382032963, 1380115619, 1362182497, 1365416520, 1366888523, 1394759973, 1377090130, 1365596761, 1388615259, 1382207581, 1365344358, 1382040163, 1365406308, 1380116069, 1393408614, 1388254140, 1394761323, 1383305324, 1368670317, 1409238638, 1360608372, 1394758773, 1412623990, 1365449236, 1380556922, 1413541653, 1354720384, 1382032513, 1381398635, 1394550721, 1404265580, 1388235402, 1362443404, 1382044813, 1380551822, 1393696403, 1362167958, 1372740759, 1377048729, 1360620406, 1368672417, 1377069730, 1381361435, 1393815204, 1381997223, 1377052329, 1383304874, 1360622686, 1380559022, 1369753629, 1379611833, 1365344958, 1366373570, 1382038213, 1359052487, 1362835659, 1365503692, 1380187170, 1360612046, 1382209231, 1394764323, 1386436308, 1380117719, 1365346008, 1372947162, 1365439439, 1408454879, 1365438689, 1365343458, 1361517795, 1360622309, 1382366423, 1383767784, 1382121313, 1380116519, 1382104813, 1380165870, 1360272253, 1369499377, 1368628467, 1394764023, 1360609017, 1366889723, 1381846314, 1397046526, 1380045569, 1373987079, 1417182337, 1354382089, 1380116819, 1362449675, 1380115319, 1381682964, 1377047829, 1406039831, 1377053979, 1377068830, 1369486627, 1377051429, 1362455848, 1382218026, 1405895979, 1380165420, 1414703405, 1377055029, 1365452086, 1394753673, 1368671967, 1416224573, 1379608383, 1388255040, 1382121013, 1398076743, 1409854795, 1380172620, 1360621389, 1381844814, 1403353423, 1362832209, 1405947730, 1377109331, 1404198228, 1371473550, 1403905878, 1382898522, 1368671067, 1383690079, 1362835809, 1362182498, 1365453286, 1396204390, 1377104232, 1360621556, 1394756973, 1365346158, 1365440374, 1408449399, 1382724472, 1380121919, 1412649340, 1381678314, 1380558722, 1389154691, 1380118919, 1362175369, 1360606602, 1398075543, 1380556172, 1361224079, 1365874576, 1377046929, 1374934124, 1382678933, 1377053079, 1394752923, 1362919837, 1362835359, 1418507680, 1377050529, 1411042211, 1403339173, 1377103782, 1362832809, 1377425834, 1381361735, 1417776046, 1360625053, 1377054129, 1365530099, 1355748791, 1374124472, 1382121913, 1380193920, 1377089980, 1382104513, 1380118169, 1367252424, 1394382071, 1366889423, 1382221776, 1374061010, 1382208931, 1380122069, 1377105881, 1380559322, 1417054684, 1362449376, 1394762673, 1364917730, 1380116219, 1374753765, 1359896550, 1360610279, 1380116969, 1365345258, 1377100781, 1376762795, 1365607411, 1388235252, 1364997374, 1403349673, 1380551672, 1394759673, 1414243498]

               for each in failure_times:
		    if event.job.submit_time > (each - 600) and event.job.submit_time < (each + 600):
				continue
		    else:	
	       """		
               waitPred = self.waitTimePredictor.get_prediction(event.job)
	       print event.job.id,
	       print "actual=\t",event.job.actual_wait_time,
	       print "queue=\t",event.job.queue_id,
               #print "\t",event.job.actual_run_time,
               #print "\t",event.job.run_time,
               #print "\t",event.job.submit_time,
	       print "pred=\t",waitPred,' '
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
    percentile_counter = counter

    for job in sorted(jobs, key=by_bounded_slow_down_sort_key):
        wait_time = float(job.start_to_run_at_time - job.submit_time)
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
