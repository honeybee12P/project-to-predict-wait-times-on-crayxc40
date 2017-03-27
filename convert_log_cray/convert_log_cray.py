import os
import datetime
import calendar
import operator
import numpy as np

def sort_table(table, col=0):
	return sorted(table, key=operator.itemgetter(col))


def convert_to_swf():
	
	job_start_time = {}
	job_user_name = {}
	job_exec_name = {}
	job_queue_name = {}
	job_run_time = {}
	job_finish_time = {}   
	job_exit_status = {}   
	job_queued_time = {}
	job_cpu_time = {}
	job_wall_time = {}
	job_mem_used = {}
	job_vmem_used = {}
	user_id_map = {}      
	unique_user_id = 1    
	exec_id_map = {}      
	unique_exec_id = 1   
	queue_id_map = {}     
	job_ncpus_used = {}   
        queue_id_map = {}
	queue_id_map['idqueue'] = 0
	queue_id_map['small'] = 1
	queue_id_map['small72'] = 2
	queue_id_map['medium'] = 3
	queue_id_map['large'] = 4
	queue_id_map['gpu'] = 5
	queue_id_map['mgpu'] = 6
	queue_id_map['xphi'] = 7
	queue_id_map['batch'] = 8
	queue_id_map['cpu_nodes'] = 9
	queue_id_map['phi_nodes'] = 10 
	queue_id_map['workq'] = 11
	path = r'/home/kruthika/Desktop/Cray Project/cray_swf_log/cray_swf/server_logs'
	user_id_cray = open("/home/kruthika/Documents/scheduling/scheduling-virtualenv/prediction/history_log_creation/user_id_cray",'w')
	server_data = {}
	for dir_entry in os.listdir(path):
		dir_entry_path = os.path.join(path, dir_entry)
		if os.path.isfile(dir_entry_path):
			with open(dir_entry_path, 'r') as my_file:
				server_data[dir_entry] = my_file.readlines()
	print 'Read', len(server_data), 'server logs'

	path = r'/home/kruthika/Desktop/Cray Project/cray_swf_log/cray_swf/sched_logs'

	old_sched_data = {}
	sched_data = {}
	for dir_entry in os.listdir(path):
		dir_entry_path = os.path.join(path, dir_entry)
		if os.path.isfile(dir_entry_path):
			with open(dir_entry_path, 'r') as my_file:
				old_sched_data[dir_entry] = my_file.readlines()
	print 'Read', len(old_sched_data), 'sched logs'

	sched_data = sorted(old_sched_data.items(),key = operator.itemgetter(0))
	
	num = 0

	for num in range(len(sched_data)):
		sched_file_data = sched_data[num][1]
		for line in sched_file_data:
			sline = line.split()
			if sline[-1] == 'run' and sline[-2].split(';')[-1] == 'Job':
				temp1 = sline[1]
				job_id = int((temp1.split('.')[0]).split(';')[-1])
				date = sline[0]
				time = sline[1].split(';')[0]
				sdate = date.split('/')
				month = int(sdate[0])
				day = int(sdate[1])
				year = int(sdate[2])
				stime = ((time.split(';'))[0]).split(':')
				hour = int(stime[0])
				mins = int(stime[1])
				sec = int(stime[2])
				startTime = datetime.datetime(year, month, day, hour, mins, sec)
				timeStamp = calendar.timegm(startTime.utctimetuple())
				job_start_time[job_id] = timeStamp
			


	job_queued_date = {}
	deleted_jobs_list = []
	batch_jobs = []
	for server_file_id in server_data:
		server_file_data = server_data[server_file_id]
		for line in server_file_data:
			sline = line.split()
			'''
			if sline[1].split(';')[5] == 'enqueuing' and sline[2] == 'into' and sline[3] == 'batch,':
				
				job_id = int((sline[1].split(';'))[4].split('.')[0])
				date = sline[0]
				time = sline[1].split(';')[0]
				sdate = date.split('/')
				month = int(sdate[0])
				day = int(sdate[1])
				year = int(sdate[2])
				stime = time.split(':')
				hour = int(stime[0])
				mins = int(stime[1])
				sec = int(stime[2])
				queuedTime = datetime.datetime(year, month, day, hour, mins, sec)
				timeStamp = calendar.timegm(queuedTime.utctimetuple())
				job_queued_time[job_id] = timeStamp
			'''
			if len(sline) > 2 and sline[2] == 'Queued':
				job_id = int((sline[1].split(';'))[4].split('.')[0])
				
				date = sline[0]
				time = sline[1].split(';')[0]
				sdate = date.split('/')
				month = int(sdate[0])
				day = int(sdate[1])
				year = int(sdate[2])
				stime = time.split(':')
				hour = int(stime[0])
				mins = int(stime[1])
				sec = int(stime[2])
				queuedTime = datetime.datetime(year, month, day, hour, mins, sec)
				timeStamp = calendar.timegm(queuedTime.utctimetuple())
				job_queued_time[job_id] = timeStamp
				
				job_user_name[job_id] = sline[6].split('@')[0]
				job_exec_name[job_id] = sline[13][:-1]
				job_queue_name[job_id] = sline[-1] 
				if sline[-1] == 'batch':
					batch_jobs.append(job_id)
					
				

			if len(sline) > 2 and sline[2].startswith('resources_used'):
				job_id = int((sline[1].split(';'))[4].split('.')[0])
				date = sline[0]
				time = sline[1].split(';')[0]
				sdate = date.split('/')
				month = int(sdate[0])
				day = int(sdate[1])
				year = int(sdate[2])
				stime = ((time.split(';'))[0]).split(':')
				hour = int(stime[0])
				mins = int(stime[1])
				sec = int(stime[2])
				finishTime = datetime.datetime(year, month, day, hour, mins, sec)
				timeStamp = calendar.timegm(finishTime.utctimetuple())
				exit_status = int(sline[1].split(';')[-1].split('=')[1])
				cput = sline[3].split('=')[1]
				cputseconds = int(cput.split(':')[0])*3600 + int(cput.split(':')[1])*60 + int(cput.split(':')[2])
				used_memory = int((sline[4].split('=')[1][:-2]))
				ncpus = int((sline[5].split('=')[1]))      
				vmem = int((sline[6].split('=')[1][:-2]))
				wallt = sline[7].split('=')[1]
				walltseconds = int(wallt.split(':')[0])*3600 + int(wallt.split(':')[1])*60 + int(wallt.split(':')[2])
				job_cpu_time[job_id] = cputseconds
				job_wall_time[job_id] = walltseconds
				job_mem_used[job_id] = used_memory
				job_finish_time[job_id] = timeStamp    
				job_vmem_used[job_id] = vmem
				job_ncpus_used[job_id] = ncpus    
				job_exit_status[job_id] = exit_status
				

			if len(sline) > 4 and sline[4] == 'deleted':
				job_id = int((sline[1].split(';'))[4].split('.')[0])
				deleted_jobs_list.append(job_id)
				
				
          
	print 'No. of job queued times read:', len(job_queued_time.keys())
	print 'No. of job in deleted list :', len(deleted_jobs_list)
	
	
	anomalies = 0
	both_Exist = 0
	anomalies = 0
	both_Exist = 0

	anom_jobs = []
	queue_max_cap = {}
	queue_max_cap['idqueue'] = 256
	queue_max_cap['small'] = 1032
	queue_max_cap['small72'] = 1032
	queue_max_cap['medium'] = 8208
	queue_max_cap['large'] = 22800
	queue_max_cap['gpu'] = 528
	queue_max_cap['mgpu'] = 500 
	queue_max_cap['xphi'] = 48

	date_boundary = datetime.datetime.today() - datetime.timedelta(seconds=86400*30)
	utc_date_boundary = calendar.timegm(date_boundary.utctimetuple())
	utc_current_time =  calendar.timegm(datetime.datetime.today().utctimetuple())

	queue_wait_time = {}
	queue_max_time = {}
	queue_max_time['idqueue'] = 86400
	queue_max_time['small'] = 86400
	queue_max_time['small72'] = 3*86400
	queue_max_time['medium'] = 86400
	queue_max_time['large'] = 86400
	queue_max_time['gpu'] = 86400
	queue_max_time['xphi'] = 86400
	queue_max_time['mgpu'] = 86400
	queue_max_time['batch'] = 86400
	queue_max_time['2'] = 86400*3

        	
	for key in job_queued_time:
		if key not in job_ncpus_used:
			job_ncpus_used[key] = None
			
	
        
	swf_log = []
	date_boundary = datetime.datetime.today() - datetime.timedelta(seconds=86400*30)
	utc_date_boundary = calendar.timegm(date_boundary.utctimetuple())
	utc_current_time =  calendar.timegm(datetime.datetime.today().utctimetuple())
	target_jobs_count = 0



	for key in job_queued_time.keys():
		        job_id = key
		        current_job = []
		        target_job = False
		      
	            		
			


	        	wait_time = 0
			if not job_id in job_start_time and job_id in job_queued_time.keys() and not job_id in deleted_jobs_list:
	            	        print "Hi inside this loop"
				wait_time = utc_current_time - job_queued_time[job_id] + 3600 # just to ensure it doesn't interfere with the discerete event simulator
	        	elif job_id in job_start_time and not job_id in deleted_jobs_list:
	            		wait_time = job_start_time[job_id] - job_queued_time[job_id]
	            		if job_start_time[key] <job_queued_time[key]:
	                		continue
			else:
				continue
			if job_ncpus_used[key] != None:
			   if queue_id_map[job_queue_name[job_id]] == 1 or queue_id_map[job_queue_name[job_id]] == 2 or queue_id_map[job_queue_name[job_id]] == 3 or queue_id_map[job_queue_name[job_id]] == 0 :
				
				current_job.append(job_id)
				current_job.append(job_queued_time[job_id])
				current_job.append(wait_time)
				if not job_id in job_start_time and job_id in job_queued_time and not job_id in deleted_jobs_list:
	            			current_job.append(0)
	        		else:
		        		if key not in job_wall_time:
	              				current_job.append(-1)
						
	            			else:
	              				current_job.append(job_wall_time[job_id])

				if not job_id in job_start_time and job_id in job_queued_time and not job_id in deleted_jobs_list or key not in job_wall_time:
					print job_id
	            			job_cpu_time[job_id] = 0
	            			job_mem_used[job_id] = 0
	            			job_exit_status[job_id] = 99999

				num_cpus = job_ncpus_used[job_id]

				
	        		avg_cpu_time = job_cpu_time[job_id]/num_cpus
				current_job.append(num_cpus)  #equivalent to nodes in tyrone cluster
				current_job.append(avg_cpu_time)
				current_job.append(job_mem_used[job_id])
				current_job.append(num_cpus)
					
				current_job.append(queue_max_time[job_queue_name[job_id]])
	        		requested_mem = -1
   				current_job.append(requested_mem)
	        		status = job_exit_status[job_id]
	        		user_id = 0
	        		if job_user_name[job_id] in user_id_map.keys():
	            			user_id =  user_id_map[job_user_name[job_id]]
	            		else:
					user_id_map[job_user_name[job_id]] = unique_user_id
	            			unique_user_id += 1
	            			user_id =  user_id_map[job_user_name[job_id]]
	        		exec_id = 0
	        		if job_exec_name[job_id] in exec_id_map.keys():
	            			exec_id =  exec_id_map[job_exec_name[job_id]]
	        		else:
	            			exec_id_map[job_exec_name[job_id]] = unique_exec_id
	            			unique_exec_id += 1
	            			exec_id =  exec_id_map[job_exec_name[job_id]]

	        		group_id = -1
	        		partition_id = -1
	        		preceeding_job = -1
	        		think_time = -1
	        		current_job.append(status)
	        		current_job.append(user_id)
	        		current_job.append(group_id)
	        		current_job.append(exec_id)
				current_job.append(queue_id_map[job_queue_name[job_id]])
	        		current_job.append(partition_id)

	        		current_job.append(preceeding_job)
	        		current_job.append(think_time)

				swf_log.append(current_job)
			
			else: 
				continue


	swf_log1 = sorted(swf_log,key = operator.itemgetter(1))

	print 'writing to swf file'
	ofile = open("/home/kruthika/Desktop/Cray Project/cray_swf_log/cray_swf/cray_log_june30.txt", 'w')
	for item in swf_log1:
		for val in item:
			print >>ofile, val, '\t\t',
		print >>ofile,""
	
	for key, value in user_id_map.iteritems():
    		user_id_cray.write(str(key))
    		user_id_cray.write("\t")
    		user_id_cray.write(str(value))
    		user_id_cray.write("\n")	

	ofile.close()
	user_id_cray.close()

if __name__ == "__main__":
	convert_to_swf()
