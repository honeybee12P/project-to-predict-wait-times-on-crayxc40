#! /usr/bin/env python2.4
#calls schedulers1/metascheduler.py which only notifies the arrival of the jobs to the predictor 
import sys
if __debug__:
    import warnings
    #warnings.warn("Running in debug mode, this will be slow... try 'python2.4 -O %s'" % sys.argv[0])

from base.workload_parser import parse_lines
from base.prototype import _job_inputs_to_jobs
from schedulers1.simulator import run_simulator
from schedulers1.metaScheduler import runMetaScheduler
import optparse
from schedulers1.fcfs_scheduler import FcfsScheduler
from schedulers1.conservative_scheduler import ConservativeScheduler
from schedulers1.double_conservative_scheduler import DoubleConservativeScheduler
from schedulers1.easy_scheduler import EasyBackfillScheduler
from schedulers1.double_easy_scheduler import DoubleEasyBackfillScheduler
from schedulers1.head_double_easy_scheduler import HeadDoubleEasyScheduler
from schedulers1.tail_double_easy_scheduler import TailDoubleEasyScheduler
from schedulers1.greedy_easy_scheduler import GreedyEasyBackfillScheduler
from schedulers1.easy_plus_plus_scheduler import EasyPlusPlusScheduler
from schedulers1.common_dist_easy_plus_plus_scheduler import CommonDistEasyPlusPlusScheduler
from schedulers1.alpha_easy_scheduler import AlphaEasyScheduler
from schedulers1.shrinking_easy_scheduler import ShrinkingEasyScheduler
from schedulers1.easy_sjbf_scheduler import EasySJBFScheduler
from schedulers1.reverse_easy_scheduler import ReverseEasyScheduler
from schedulers1.perfect_easy_scheduler import PerfectEasyBackfillScheduler
from schedulers1.double_perfect_easy_scheduler import DoublePerfectEasyBackfillScheduler
from schedulers1.lookahead_easy_scheduler import LookAheadEasyBackFillScheduler
from schedulers1.orig_probabilistic_easy_scheduler import OrigProbabilisticEasyScheduler
from schedulers1.log_scheduler import LogScheduler
def main():
    
    #input_file = '/home/sharath/Desktop/pyss-read-only/src/5K_sample'
    input_file = '/home/sharath/Desktop/pyss-read-only/src/Tyrone_log'
    num_processors1 = 800
    input_file = open(input_file)
    
    scheduler = LogScheduler(num_processors1)
    #scheduler = EasyBackfillScheduler(num_processors1)

    try:
        print "...." 
        runMetaScheduler(
                num_processors = num_processors1, 
                jobs = _job_inputs_to_jobs(parse_lines(input_file), num_processors1),
                scheduler = scheduler 
            )

        #print "Num of Processors: ", options.num_processors
        #print "Input file: ", options.input_file
        #print "Scheduler:", type(scheduler)
        #print 'Total scheduled jobs:', scheduler.totalScheduledJobs
    finally:
        if input_file is not sys.stdin:
            input_file.close()


if __name__ == "__main__":# and not "time" in sys.modules:
    try:
        import psyco
        psyco.full()
    except ImportError:
        print "Psyco not available, will run slower (http://psyco.sourceforge.net)"
    main()
