from common import Scheduler, CpuSnapshot
from base.prototype import JobStartEvent

class LogScheduler(Scheduler):

    def __init__(self, num_processors):
        super(LogScheduler, self).__init__(num_processors)
        self.cpu_snapshot = CpuSnapshot(num_processors)
        self.waiting_queue_of_jobs = []

    def new_events_on_job_submission(self, job, current_time):
        self.cpu_snapshot.archive_old_slices(current_time)
	result = []        
	#self.waiting_queue_of_jobs.append(job)
	result.append(JobStartEvent(current_time+job.actual_wait_time, job))
        #return [
        #    JobStartEvent(current_time, job)
        #    for job in self._log_schedule_jobs(current_time)
        #]
	return result

    def new_events_on_job_termination(self, job, current_time):
        self.cpu_snapshot.archive_old_slices(current_time)
        self.cpu_snapshot.delTailofJobFromCpuSlices(job)
	return []        
	"""return [
            JobStartEvent(current_time, job)
            for job in self._log_schedule_jobs(current_time)
        ]"""


    def _schedule_jobs(self, current_time):
        result = []
        while len(self.waiting_queue_of_jobs) > 0:
            job = self.waiting_queue_of_jobs[0]
            if self.cpu_snapshot.free_processors_available_at(current_time) >= job.num_required_processors:
                self.waiting_queue_of_jobs.pop(0)
                self.cpu_snapshot.assignJob(job, current_time)
                result.append(job)
            else:
                break
        return result
    
    def _log_schedule_jobs(self, current_time):                 #by Siddharth
        result = []
	return result
        """while len(self.waiting_queue_of_jobs) > 0:
            job = self.waiting_queue_of_jobs[0]
            if job.submit_time + job.actual_wait_time <= current_time:
                self.waiting_queue_of_jobs.pop(0)
                self.cpu_snapshot.assignJob(job, current_time)
                result.append(job)
            else:
                break
        return result"""
