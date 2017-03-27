import math, sys, os
from time import clock, time
from operator import mul
import numpy
import random
from scipy.stats import pearsonr
import scipy
import scipy.cluster.vq
import scipy.spatial.distance
#from sklearn.cluster import Kmeans
dst = scipy.spatial.distance.euclidean


from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from sklearn.cluster import dbscan


from sklearn import linear_model 
from numpy.linalg import cond as COND_NUMBER
np = numpy


np.random.seed(9876789)

check_list  = []

################################################################
##########    PARAMETERS FOR SENSITIVITY ANALYSIS    ###########
################################################################

#test_trace = "Tyrone"
test_trace = "cray"
#test_history_size = 5000
test_history_size = 1000          #history size edited while working for cray 
#test_target_start_point_in_trace = 5000
test_target_start_point_in_trace = 41184
#TARGET_SIZE = 30000 
TARGET_SIZE = 3650

# Clustering defn
test_epsilon = 0.01  
test_minpts = 3

# SD Minimizer
test_top_x_percent = 1.0/100.0
test_distance_threshold = 0.4
test_step_size = 0.04

test_outlier_fraction = 0.3
#test_growing_window is used for this graphs
test_growing_window = False  # False is fixed size moving window
test_remove_outlier = True

# Distance function
test_use_distribution = False
test_use_weights_for_distributions = True

# Other methods
test_regression_distance = 0.6
test_weighted_avg_neighbour_count = 1
test_ridge_regularization_constant = 0.1
################################################################
################################################################
#result_file = open("/home/siddharthsahu/Desktop/Validation_job/Validation_modified_time_art_2/web_source/web_result_art",'w')
ridge_reg_model = linear_model.Ridge(alpha=test_ridge_regularization_constant)

def fastsort(a):
    """
    Sort an array and provide the argsort

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    fastsort : ndarray of type int
        sorted indices into the original array

    """
    # TODO: the wording in the docstring is nonsense.
    it = np.argsort(a)
    as_ = a[it]
    return as_, it

def rankdata(a):
    """
    Ranks the data, dealing with ties appropriately.

    Equal values are assigned a rank that is the average of the ranks that
    would have been otherwise assigned to all of the values within that set.
    Ranks begin at 1, not 0.

    Parameters
    ----------
    a : array_like
        This array is first flattened.

    Returns
    -------
    rankdata : ndarray
         An array of length equal to the size of `a`, containing rank scores.

    Examples
    --------
    >>> stats.rankdata([0, 2, 2, 3])
    array([ 1. ,  2.5,  2.5,  4. ])

    """
    a = np.ravel(a)
    n = len(a)
    svec, ivec = fastsort(a)
    sumranks = 0
    dupcount = 0
    newarray = np.zeros(n, float)
    for i in xrange(n):
        sumranks += i
        dupcount += 1
        if i==n-1 or svec[i] != svec[i+1]:
            averank = sumranks / float(dupcount) + 1
            for j in xrange(i-dupcount+1,i+1):
                newarray[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newarray

def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis
    return a, outaxis
def spearmanr(a, b=None, axis=0):
    a, axisout = _chk_asarray(a, axis)
    ar = np.apply_along_axis(rankdata,axisout,a)

    br = None
    if not b is None:
        b, axisout = _chk_asarray(b, axis)
        br = np.apply_along_axis(rankdata,axisout,b)
    n = a.shape[axisout]
    rs = np.corrcoef(ar,br,rowvar=axisout)

    olderr = np.seterr(divide='ignore')  # rs can have elements equal to 1
    try:
        t = rs * np.sqrt((n-2) / ((rs+1.0)*(1.0-rs)))
    finally:
        np.seterr(**olderr)

    if rs.shape == (2,2):
        return rs[1,0], 0.001
    else:
        return rs, 0.001

    return rs,0.001
max_cluster_count = 5
def gap(data, refs=None, nrefs=20, ks=range(1,max_cluster_count)):
    shape = data.shape
    if refs==None:
        tops = data.max(axis=0)
        bots = data.min(axis=0)
        dists = scipy.matrix(scipy.diag(tops-bots))
 
        rands = scipy.random.random_sample(size=(shape[0],shape[1],nrefs))
        for i in range(nrefs):
            rands[:,:,i] = rands[:,:,i]*dists+bots
    else:
        rands = refs
 
    gaps = scipy.zeros((len(ks),))
    for (i,k) in enumerate(ks):
        (kmc,kml) = scipy.cluster.vq.kmeans2(data, k)
        disp = sum([dst(data[m,:],kmc[kml[m],:]) for m in range(shape[0])])
 
        refdisps = scipy.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            (kmc,kml) = scipy.cluster.vq.kmeans2(rands[:,:,j], k)
            refdisps[j] = sum([dst(rands[m,:,j],kmc[kml[m],:]) for m in range(shape[0])])
        gaps[i] = scipy.log(scipy.mean(refdisps))-scipy.log(disp)
    return gaps

class Plain_Weighted_Predictor:
	def __init__(self, id_value):
		self.attr_wts = []
		self.predictor_id = id_value
		self.neighbour_size = id_value
		self.my_predictions = {}
		#print 'Setting up WA predictor,', self.predictor_id, 'K=', self.neighbour_size

	def get_prediction_plain(self, individual_distances, individual_wait_times, job_id, weights):
		self.attr_wts = weights
		self.prediction_list = []
		self.distance_map = {}
                if len(individual_distances) == 0:
                    return 0

		for key in individual_distances.keys():
			distance = individual_distances[key]
			current_wait = individual_wait_times[key]
			temp_overall_distance_num = temp_overall_distance_deno = 0
			for k in range(len(distance)):
				temp_overall_distance_num += (self.attr_wts[k]*distance[k]*distance[k])
				temp_overall_distance_deno += self.attr_wts[k]
			overall_distance = float(temp_overall_distance_num)/float(temp_overall_distance_deno)
			overall_distance = math.sqrt(overall_distance)
			self.distance_map[key] = overall_distance
			self.prediction_list.append( (overall_distance, current_wait, key) )

			
		self.prediction_list.sort()

		### K-NN WA ###
		if self.neighbour_size == 1:
                        if len(self.prediction_list):
                            predicted_wait = int(self.prediction_list[0][1])
                        else:
                            predicted_wait = 0
		else:
			count = 0 ; temp_num = 0 ; temp_deno = 0
			for k in range(len(self.prediction_list)):
				temp_num += (math.exp(-(self.distance_map[self.prediction_list[k][2]]))*self.prediction_list[k][1])
				temp_deno += (math.exp(-(self.distance_map[self.prediction_list[k][2]])))
				count += 1
				if count == self.neighbour_size:
					break
			predicted_wait = int(temp_num/temp_deno)
		self.my_predictions[job_id] = predicted_wait
                if job_id == 53706:
			print "validation:\t","prediction for 53706 coming from Plain weighted predictor\t",predicted_wait
		return predicted_wait

	def get_score(self, correct_predictions):
		if len(correct_predictions.keys()) < 5:
			return 1.0
		score = 0.0
		count = 0.0
		for key in correct_predictions.keys():
			if key in self.my_predictions.keys():
				score += abs(correct_predictions[key] - self.my_predictions[key])
				count += 1
		return score/count

class DBSCAN_Based_SDScan:
        # For DBSCAN clustering
        def var_wait_graph_vector_distance(self, X, Y):    
            '''
            X and Y are two dimensional
            '''
            distance = float(X[0] - Y[0])/self.qwt_range + float(X[1] - Y[1])/1.0
            #print distance
            return distance

        # For DBSCAN clustering
        def var_wait_graph_distance(self, X_ARR, Y_ARR):
            
            if len(X_ARR) == 2 and len(Y_ARR) == 2:
                X = X_ARR; Y = Y_ARR;
                distance = abs(float(X[1] - Y[1]))/self.qwt_range + abs(float(X[0] - Y[0]))
                #print distance
                return distance

            D = numpy.zeros([len(X_ARR), len(Y_ARR)])
            for iX in range(len(X_ARR)):
                for iY in range(len(Y_ARR)):
                    D[ix][iY] = self.var_wait_graph_vector_distance(X_ARR[iX], Y_ARR[iY])
            return D

        def var_wait_graph_distance_matrix(self, X_ARR):
            D = self.var_wait_graph_distance(X_ARR, X_ARR)
            return D

	def __init__(self, id_value, wait_time_range):
            self.attr_wts = []
            self.predictor_id = id_value
            self.my_predictions = {}
            self.qwt_range = wait_time_range
            #print self.qwt_range
            #print 'Setting up DBSCAN based s.d.minimizer'
            self.density_clustering = DBSCAN(eps=test_epsilon, min_samples=test_minpts, metric=self.var_wait_graph_distance)
            self.pick_global_outliers_clustering = DBSCAN(eps=0.1, min_samples=5, metric=self.var_wait_graph_distance)

        ##
        # get_global_outliers assumes self.prediction_list is updated and correct, call ONLY after doing get_prediction_plain. 
        ##            
        def get_global_outliers(self):
            x_percent = 1.0
            top_x_percent = int(x_percent*len(self.prediction_list))
            #print "top_x_percent from get_global_outliers", top_x_percent
            count_x_percent = 0
            current_weights_list = []
            sample_array = []
            job_list = []
            while count_x_percent < top_x_percent:
                item = self.prediction_list[count_x_percent]
                sample_array.append([item[0], item[1]])
                job_list.append(item[2])
                count_x_percent += 1
	    if job_id == 53706:
	    	print "validation:\t",job_id,"\t","Similar jobs coming from DBScan get global outliers\t",job_list
            np_sample_array = numpy.array(sample_array)
            db = self.density_clustering.fit(np_sample_array)
            core_samples = db.core_sample_indices_
            labels = db.labels_
            X = np_sample_array
            outlier_list = []
            outlier_count = sum(1 for l in labels if l==-1)  

            for (l, position, id_value) in zip(labels, sample_array, job_list):
                if l == -1:
                    outlier_list.append(id_value)
            print 'Found', len(outlier_list), 'global outliers'
            return outlier_list
        def preprocess_for_regression(self, individual_distances, individual_wait_times, job_id, weights,wait_for_plotting):
                self.attr_wts = weights
		self.prediction_list = []
		self.distance_map = {}
		for key in individual_distances.keys():
			distance = individual_distances[key]
			current_wait = individual_wait_times[key]
			temp_overall_distance_num = temp_overall_distance_deno = 0
			for k in range(len(distance)):
				temp_overall_distance_num += (self.attr_wts[k]*distance[k])
				temp_overall_distance_deno += self.attr_wts[k]
                        
			overall_distance = float(temp_overall_distance_num)/float(temp_overall_distance_deno)
			overall_distance = math.sqrt(overall_distance)
			self.distance_map[key] = overall_distance
			self.prediction_list.append((overall_distance, current_wait, key))
		if job_id == 53706:
	        	print "validation:\t",job_id,"\t Prediction list from dbscan regress\t",self.prediction_list
		self.prediction_list.sort()
                return self.prediction_list, self.distance_map
	#get_prediction_function
       	def get_prediction_plain(self, individual_distances, individual_wait_times, job_id, weights,wait_for_plotting):
		self.attr_wts = weights   #weights of distributions calculated by spearman's
		self.prediction_list = []
		self.distance_map = {}
		if job_id == 53706:
		    print "validation: keys=",individual_distances.keys()
		    print "validation: attr_wts=",self.attr_wts
		for key in individual_distances.keys():
			distance = individual_distances[key]
			current_wait = individual_wait_times[key]
			temp_overall_distance_num = temp_overall_distance_deno = 0
			#validation
			if job_id == 53706:
			    print "validation: job_id=",job_id," key=",key
			    print "validation: keys=",individual_distances.keys()
			    print "validation: distance=",distance
			    print "validation: attr_wts=",self.attr_wts
			for k in range(len(distance)):
				temp_overall_distance_num += (self.attr_wts[k]*distance[k])   #for equation 2 in pt. 3.2.1 in paper
				temp_overall_distance_deno += self.attr_wts[k]
                        
			overall_distance = float(temp_overall_distance_num)/float(temp_overall_distance_deno)
			overall_distance = math.sqrt(overall_distance)
			self.distance_map[key] = overall_distance
			self.prediction_list.append((overall_distance, current_wait, key))

			
		self.prediction_list.sort() #this will sort on first value i.e. overall_distance

                if len(self.prediction_list) == 0:
                    return 0
                
                wa_flag = False

                mindist = self.prediction_list[0][0]

                # Take top 1% points in the array and do DBSCAN for those only.
                x_percent = test_top_x_percent
                DISTANCE_THRESHOLD = test_distance_threshold

                top_x_percent = int(x_percent*len(self.prediction_list))+1
                count_x_percent = 0
                current_weights_list = []
                top_distance_list = []
                sample_array = []
                job_list = []
                #print job_list
                while count_x_percent < top_x_percent:
                    item = self.prediction_list[count_x_percent]
                    top_distance_list.append(item[0])
                    #                    distance, wait_time
                    sample_array.append([item[0], item[1]])
                    job_list.append(item[2])
                    count_x_percent += 1
		if job_id == 53706:
                	print "validation:\t",job_id,"\tSimilar Jobs from dbscan of get prediction plain\t",job_list
                np_sample_array = numpy.array(sample_array) #sample_array has tuples(overall_distance, wait_time) for top_x_percent jobs
                #print job_id, 'dbscan input size:', len(sample_array)
                #print job_id,"Top distance list\t",top_distance_list
                db = self.density_clustering.fit(np_sample_array)  # for help of DBSCAN	http://madhukaudantha.blogspot.in/2015/04/density-based-clustering-algorithm.html
                core_samples = db.core_sample_indices_        
                labels = db.labels_
                X = np_sample_array
                #print labels
                outlier_list = []
                outlier_count = sum(1 for l in labels if l==-1)  

                for (l, position, id_value) in zip(labels, sample_array, job_list):
                    if l == -1:
                        outlier_list.append(id_value)
                
                if len(outlier_list) != outlier_count:
                    
                    with open('n.txt','a') as f2:	
                                f2.write("{0}\n".format("Problem with outlier count!"))
                else:
                    outlier_count = len(outlier_list)

                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

                unique_labels = set(labels)
                count_list = []
                #class_distance_list = []
                for k in unique_labels:
                    
                    class_members = [index[0] for index in np.argwhere(labels == k)]
                    #print job_id,"CLASS",class_members   
                    class_distance = [X[index][0] for index in class_members]
                    #print zip(class_members, class_distance)
                    #class_distance_list.append(numpy.mean(class_distance))
                    count_list.append(len(class_members))

                avg_cluster_size = float(sum(count_list[:-1]))/len(count_list)
                avg_distance_top = numpy.mean(top_distance_list)
                #print top_distance_list
                #avg_class_distance = numpy.mean(class_distance_list)
                if n_clusters_ == 1:
                    avg_cluster_size = top_x_percent
                print ""
                print job_id, 'Outliers:', outlier_count, 'Clusters:', n_clusters_, 'Points:', top_x_percent, 'avg_cluster_size:', avg_cluster_size, 'avg_distance_top:', avg_distance_top#, 'avg_class_distance:', avg_class_distance
                print ""
                
                #print 'Points:', len(current_window_wait)
                #print('Estimated number of clusters: %d' % n_clusters_)
                DISPLAY_GRAPH = False
                if DISPLAY_GRAPH:
                    colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
                    for k, col in zip(unique_labels, colors):
                        if k == -1:
                            # Black used for noise.
                            col = 'k'
                            markersize = 15
                        class_members = [index[0] for index in np.argwhere(labels == k)]
                        cluster_core_samples = [index for index in core_samples
                                                if labels[index] == k]
                        for index in class_members:
                            x = X[index]
                            if index in core_samples and k != -1:
                                markersize = 7
                            else:
                                markersize = 6
                            plt.plot(x[0], x[1]-wait_for_plotting, 'o', markerfacecolor=col,
                                    markeredgecolor='k', markersize=markersize)

                    #plt.title('Estimated number of clusters: %d' % n_clusters_)
                    plt.xlim(xmin=0, xmax=1.0)
                    plt.ylim(ymin=-200000, ymax=400000)
                    plt.title(str(job_id))

                    #plt.axvline(x=current_dist, ls='dashed', color='y')
                    plt.show()
                    #return 0,self.prediction_list
                
                prev_outlier_count = outlier_count
                # Conditions for CTC - True => cluster structure is good. Go ahead! Otherwise, try regression.
                go_ahead = True
                #if top_x_percent < 10:
                #    go_ahead = True
                is_suitable_regression = False
                predicted_wait = 0.0
                if outlier_count < test_outlier_fraction*avg_cluster_size:
           
                    print 'Not many outliers', outlier_count
                go_ahead = ((outlier_count == 0) and (avg_distance_top < DISTANCE_THRESHOLD + 0.1)) or ((avg_distance_top <= DISTANCE_THRESHOLD and outlier_count < test_outlier_fraction*avg_cluster_size)) #Sec 3.3.1 equation 5
                print 'Go ahead with sd scanning:', go_ahead
                if go_ahead:
                    ## need to find a new outlier list for all jobs till 0.5 distance
                    if max(top_distance_list) < DISTANCE_THRESHOLD:
                        # add the remaining points to sample_list and cluster again to get the correct list of outliers
                        for item in self.prediction_list:
                            if item[0] <= max(top_distance_list) or item[0] > DISTANCE_THRESHOLD:
                                continue
                            sample_array.append([item[0], item[1]])
                            job_list.append(item[2])

                        np_sample_array = numpy.array(sample_array)
                        X = np_sample_array
                        #print job_list
			if job_id == 53706:
                		print "validation:\t",job_id,"\tSimilar Jobs from dbscan of get prediction plain (second part down)\t",job_list
                        #print job_id, 'dbscan input size:', len(sample_array)
                        db = self.density_clustering.fit(np_sample_array)
                        core_samples = db.core_sample_indices_
                        labels = db.labels_

                        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

                        unique_labels = set(labels)
                        #print labels
                        outlier_list = []
                        for (l, position, id_value) in zip(labels, sample_array, job_list):
                            if l == -1:
                                outlier_list.append(id_value)
                        if prev_outlier_count < len(outlier_list):
                            print len(outlier_list)-prev_outlier_count, 'outliers found'
                            with open('n.txt','a') as f2:	
                                f2.write("{0}\n".format("outliers found"))
                        DISPLAY_BIG_GRAPH = False
                        if DISPLAY_BIG_GRAPH:
                            plt.clf()
                            colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
                            for k, col in zip(unique_labels, colors):
                                if k == -1:
                                    # Black used for noise.
                                    col = 'k'
                                    markersize = 15
                                class_members = [index[0] for index in np.argwhere(labels == k)]
                                cluster_core_samples = [index for index in core_samples
                                                        if labels[index] == k]
                                for index in class_members:
                                    x = X[index]
                                    if index in core_samples and k != -1:
                                        markersize = 7
                                    else:
                                        markersize = 6
                                    plt.plot(x[0], x[1]-wait_for_plotting, 'o', markerfacecolor=col,
                                            markeredgecolor='k', markersize=markersize)

                            #plt.title('Estimated number of clusters: %d' % n_clusters_)
                            plt.xlim(xmin=0, xmax=1.0)
                            plt.ylim(ymin=-200000, ymax=400000)
                            plt.title(str(job_id))
                            #plt.axvline(x=current_dist, ls='dashed', color='y')
                            plt.show()
                    #print 'Condition satisfied'
                    current_dist = mindist
                    window_size = test_step_size
                    maxdist = DISTANCE_THRESHOLD-window_size

                    sd_list = []
                    point_count = []                
                    iter_list = []
                    iter_count = 0
                    window_list = []
                    weights_list = []
                    max_non_zero_list_pos = 0
                    pred_list = []
                    pos_index = []
                    while current_dist < maxdist:
                        iter_count += 1
                        # pick points in current_dist + window_size window
                        current_window_wait = []
                        current_weights_list = []
                        #sample_array = []
                        #print 'Window', current_dist, 'to', current_dist + window_size
                        #print len(self.prediction_list)
                        for item in self.prediction_list:
                            # get a cluster with outliers eliminated
                            if test_remove_outlier:
                                if test_growing_window:
                                    if item[0] >= mindist and item[0] < current_dist + window_size and (not item[2] in outlier_list): 
                                        current_window_wait.append(item[1])
                                        current_weights_list.append(math.exp(-item[0]*item[0]))
                                else:
                                    if item[0] >= current_dist and item[0] < current_dist + window_size and (not item[2] in outlier_list): 
                                        current_window_wait.append(item[1])
                                        current_weights_list.append(math.exp(-item[0]*item[0]))
                            else:
                                if test_growing_window:
                                    if item[0] >= mindist and item[0] < current_dist + window_size: 
                                        current_window_wait.append(item[1])
                                        current_weights_list.append(math.exp(-item[0]*item[0]))
                                else:
                                    if item[0] >= current_dist and item[0] < current_dist + window_size: 
                                        current_window_wait.append(item[1])
                                        current_weights_list.append(math.exp(-item[0]*item[0]))

                        if 0 and len(sample_array) == 0:
                            plt.axvline(x=current_dist, ls='dashed', color='y')
                            current_dist += window_size
                            continue
                        #np_sample_array = numpy.array(sample_array)

                        #db = self.density_clustering.fit(np_sample_array)
                        #core_samples = db.core_sample_indices_
                        #labels = db.labels_
                        #X = np_sample_array

                        #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                        #print 'Points:', len(current_window_wait)
                        #print('Estimated number of clusters: %d' % n_clusters_)
                        '''
                        unique_labels = set(labels)
                        colors = pl.cm.Spectral(np.linspace(0, random.random(), len(unique_labels)))
                        for k, col in zip(unique_labels, colors):
                            if k == -1:
                                # Black used for noise.
                                col = 'k'
                                markersize = 15
                            class_members = [index[0] for index in np.argwhere(labels == k)]
                            cluster_core_samples = [index for index in core_samples
                                                    if labels[index] == k]
                            for index in class_members:
                                x = X[index]
                                if index in core_samples and k != -1:
                                    markersize = 7
                                else:
                                    markersize = 6
                                plt.plot(x[0], x[1]-wait_for_plotting, 'o', markerfacecolor=col,
                                        markeredgecolor='k', markersize=markersize)

                        #plt.title('Estimated number of clusters: %d' % n_clusters_)
                        plt.xlim(xmin=0, xmax=1.0)
                        plt.ylim(ymin=-100000, ymax=400000)

                        plt.axvline(x=current_dist, ls='dashed', color='y')
                        #plt.show()


                        if len(current_window_wait) > 0:
                            max_non_zero_list_pos += 1
                        '''
                        current_sd = numpy.std(current_window_wait)
                        #iter_list.append(iter_count)
                        if current_sd == 0.0 or len(current_window_wait) <= 1:
                            current_sd = float('inf')

                        sd_list.append(current_sd)
                        point_count.append(len(current_window_wait))
                        window_list.append(current_window_wait)
                        weights_list.append(current_weights_list)
                        #if len(current_window_wait) > 1:
                        #    #pred_list.append(numpy.average(window_list[-1], weights=weights_list[-1]))
                        #    pred_list.append(numpy.mean(window_list[-1]))
                        #else:
                        #    pred_list.append(-1)
                        pos_index.append(current_dist+window_size)
                        current_dist += window_size
                    #plt.show()

                    #min_pos = sd_list.index(min(sd_list[:min(10,max_non_zero_list_pos)]))
                    if job_id in check_list:
                        #print sd_list//
                        with open('n.txt','a') as f2:	
                                f2.write("{0}\n".format(sd_list))
                        #print point_count
                        #print pred_list
                        #print pos_index
                    if len(sd_list) == 0:
                        #print 'sd_list empty!'
                        return -1, self.prediction_list, self.distance_map, True
                    if min(sd_list) == float('inf'):
                        #print 'sd_list minima is inf!'
                        return -1, self.prediction_list, self.distance_map, True
                    min_pos = sd_list.index(min(sd_list))
                    #print 'min_pos:', min_pos
                    #print window_list[min_pos], weights_list[min_pos]
                    predicted_wait = numpy.average(window_list[min_pos], weights=weights_list[min_pos]) #provides solution of SDM
                    if job_id == 53706:
			print "validation\t","prediction for 41235 coming from DBScan\t",predicted_wait
                    return predicted_wait, self.prediction_list, self.distance_map,False
                else:	#if not go ahead in line 651 then return -1 in place of result i.e. predict_wait
                    print 'Returning -1', len(self.prediction_list), len(self.distance_map)
                    return -1, self.prediction_list, self.distance_map,True # True means 'is suitable for regression'

	def get_score(self, correct_predictions):
		if len(correct_predictions.keys()) < 5:
			return 1.0
		score = 0.0
		count = 0.0
		for key in correct_predictions.keys():
			if key in self.my_predictions.keys():
				score += abs(correct_predictions[key] - self.my_predictions[key])
				count += 1
		return score/count


class Batch_System:
	
	# Initialize all the variables
	def __init__(self):
		self.__History_Begin = test_target_start_point_in_trace-test_history_size; #begining of history
                if len(check_list):         #if check list is not empty then test target point is check list
                    self.__History_Begin = (check_list[0]-4002)   #history begins 4002 earlier
                self.__History_End = test_target_start_point_in_trace ;   #end of history
                if len(check_list):
                    self.__History_End = check_list[0]-1
                self.__TargetJobs_Begin = self.__History_End + 1 ; self.__Target_Size = TARGET_SIZE
                
                    
		self.Max_Nodes 	= 0 ; self.__free_nodes = self.Max_Nodes ; self.__Near_Mid_ShortJobs = 0
		self.__HISTORY_SIZE = self.__History_End - self.__History_Begin
		self.__good_metric_ibl = 0 ; 
		self.__history_jobs = [] ; self.__target_jobs = []
		self.__waiting_job_id = [] ; self.__waiting_job_ert = [] ; self.__waiting_job_reqsize = []  ; self.__waiting_job_cpu = []
		self.__running_job_id = [] ; self.__running_job_reqsize = [] ; self.__running_job_ert = [] ; self.__running_job_cpu = []
		self.__arr_time = {} ; self.__start_time = {} ; self.__wait_time = {} ; self.__real_run_time = {} ; self.__req_size = {} ; self.__est_runtime = {} ; 
		self.__queue_id = {} ; self.__queue_group = {} ; self.__user_id = {} ; self.__grp_id = {} ; self.__exec_id = {}
		self.__job_rank_reqsize_map = {} ; self.__qsize_map = {} ; self.__free_node_map = {} ; self.__proc_occ_map = {} ; self.__job_rank_ert_map = {} ; self.__job_rank_combo_map = {}
		self.__prev_req_waitq = {} ; self.__prev_ert_waitq = {} ; self.__job_rank_threshold = {}		
		self.__metric_list_ibl = [] ; self.__metric_list_ibl_map = {}
		self.__ibl_predicted_wait_time = {} ; 
		self.__proc_occ_reqsize_rms = {} ; self.__proc_occ_ert_rms = {} ; self.__queue_occ_reqsize_rms = {} ; self.__queue_occ_ert_rms = {} 
		self.__proc_occ_reqsize_avg = {} ; self.__proc_occ_ert_avg = {} ; self.__proc_occ_cputime_avg = {}
		self.__proc_occ_no_jobs = {} ; self.__queue_occ_no_jobs = {}; self.__proc_occ_elapsed_time_total = {} ; self.__proc_occ_remaining_time_total = {} ; self.__queue_occ_elapsed_time_total = {} ; self.__queue_occ_demand_time_total = {}
		self.__queue_occ_reqsize_avg = {} ; self.__queue_occ_ert_avg = {} ; self.__queue_occ_cputime_avg = {}
		self.record_waiting_jobs = {}
		self.record_running_jobs = {}
		self.PredictedWaitingTime = {}

		# Submitting user based features 
		self.user_queue_waiting_jobs = {}
		self.user_queue_demand_wallclock_sum = {}
		self.user_queue_demand_reqsize_sum = {}
		self.user_queue_demand_ert_sum = {}

                self.user_queue_elapsed_wallclock_sum = {}

 		self.user_queue_features_min = [float('inf')]*5
		self.user_queue_features_max = [-float('inf')]*5

		self.user_proc_running_jobs = {}
		self.user_proc_demand_wallclock_sum = {}
		self.user_proc_demand_reqsize_sum = {}
		self.user_proc_demand_ert_sum = {}
                self.user_proc_elapsed_wallclock_sum = {}

		self.user_proc_features_min = [float('inf')]*5
		self.user_proc_features_max = [-float('inf')]*5

                self.sum_wait_ert = {}
                self.sum_wait_req = {}
                self.sum_wait_elapsed = {}
                self.sum_run_ert = {}
                self.sum_run_req = {}
                self.sum_run_elapsed = {}
		self.sum_higher_priority_jobs = {} 	#cray_addition_sid
		self.sum_starving_jobs = {}		#cray_addition_sid
		#self.queue_waiting_jobs = {}		#cray_addition_sid
		#self.queue_running_jobs = {}		#cray_addition_sid
                
                #self.sum_test_min = [0.0]*6
                self.sum_test_min = [float('inf')]*8
                self.sum_test_max = [-float('inf')]*8
                self.queue_id_map = {}
		"""
                self.queue_id_map[0] = 'small72'
                self.queue_id_map[1] = 'small'
                self.queue_id_map[2] = 'medium'
                self.queue_id_map[3] = 'large'
                self.queue_id_map[4] = 'gpu'
                self.queue_id_map[4] = 'xphi'
		self.queue_id_map[4] = 'idqueue'
		"""
		self.queue_id_map[0] = 'idqueue'
		self.queue_id_map[1] = 'small'
		self.queue_id_map[2] = 'small72'
		self.queue_id_map[3] = 'medium'
		self.queue_id_map[4] = 'large'
		self.queue_id_map[5] = 'gpu'
		self.queue_id_map[6] = 'mgpu'
		self.queue_id_map[7] = 'xphi'
		self.queue_id_map[8] = 'batch'
		self.queue_id_map[9] = 'cpu_nodes'
		self.queue_id_map[10] = 'phi_nodes' 
                # Prakash: Can we add user queue elapsed wallclock time and user proc elapsed wall clock time??

                #self.UNWEIGHTED = False
		self.ENABLE_DISTRIBUTIONS = test_use_distribution
                self.ENABLE_SUMMARY_FEATURES = True
                self.ENABLE_SUMMARY_FEATURES_FOR_REGRESSION = True
                
                #if self.ENABLE_DISTRIBUTIONS:
                    #print 'Distribution logic'
                    #with open('n.txt','a') as f2:	
                                #f2.write("{0}\n".format("Distribution logic"))
                #if self.ENABLE_SUMMARY_FEATURES:
                    #print 'Feature logic'
                    #with open('n.txt','a') as f2:	
                                #f2.write("{0}\n".format("Feature logic"))
		self.bin_count = 10

		# Features weights
		# NOMINAL & NUMBER - JOB ATTR.
		WGROUP_ID = 0
		WUSER_ID = 0
		WQUEUE_ID = 0
		WEXEC_ID = 0
		WREQ_SIZE = 1
		WERT = 1
		WARR_TIME = 1
		WFREE_NODES = 1

		# VECTOR
		WQUEUE_DEMAND_VECTOR = 1
		
		# DISTRIBUTIONS
		WJW_ERT = 1		#all there variables can take value as 0 or 1
		WJW_REQ_SIZE = 1
		WJW_ELAPSED = 1
		WJW_WALLCLOCK = 0
		
		WJR_ERT = 1		
		WJR_REQ_SIZE = 1
		WJR_ELAPSED = 1
		WJR_WALLCLOCK = 0

		# NUMERIC
		WUQ_COUNT = 1
		WUQ_DEMAND_WALLCLOCK = 1
		WUQ_DEMAND_REQ_SIZE = 1
		WUQ_DEMAND_ERT = 1

		self.selected_distribution_list = [WJW_ERT, WJW_REQ_SIZE, WJW_ELAPSED, WJW_WALLCLOCK, WJR_ERT, WJR_REQ_SIZE, WJR_ELAPSED, WJR_WALLCLOCK]	#values having 1 represent distribution included and zero represents not included
                
		self.__attr_wts =  [WREQ_SIZE, WERT, WARR_TIME, WFREE_NODES]
		self.__attr_wts += [WQUEUE_DEMAND_VECTOR]

		for i in range(len(self.selected_distribution_list)):
			if self.selected_distribution_list[i] != 0:
				self.__attr_wts.append(self.selected_distribution_list[i])

		self.__attr_wts += [WUQ_COUNT, WUQ_DEMAND_WALLCLOCK, WUQ_DEMAND_REQ_SIZE, WUQ_DEMAND_ERT]
		
		
		self.__neighbout_size = 2

		#### PREDICTOR STRUCTURE ####
		# Each predictor is defined by a set of len(self.__attr_wts), and number k which is the number of neighbours
		# For each predictor, the success is measured by the absolute error
		# Given a job, pick the predictor with the best absolute error
		##############################

		self.Predictor_Length = len(self.__attr_wts)
		self.Predictor_Count = 4

                self.WA_Predictor = Plain_Weighted_Predictor(test_weighted_avg_neighbour_count)
                #print self.WA_Predictor
		self.Score_Round_Frequency = 5
		self.Jobs_Since_Last_Score_Computation = 0
		self.Jobs_for_score = {}
		self.__ert_range = 0.0; self.__arr_range = 0.0 # these two parameters are found dynamically

		self.__ert_max = 0.0
		self.__ert_min = float('inf')
		self.__arr_max = 0.0
		self.__arr_min = float('inf')
                self.__req_max = 0.0
                self.__req_min = float('inf')
                self.__elapsed_wait_max = 0.0
                self.__elapsed_wait_min = 0.0
                self.__elapsed_run_max = 0.0
                self.__elapsed_run_min = 0.0

                self.job_feature_vector = {}
 
                if test_trace == 'Tyrone':
                    self.__no_queues = 6
                    self.train_ert_min = 7200
                    self.train_ert_max = 86400
                    self.train_req_min = 32.0
                    self.train_req_max = 256.0
                    self.train_run_min = 1.0
                    self.train_run_max = 86000.0
                    self.train_wait_min =1.0
                    self.train_wait_max = 86000.0
		
		if test_trace == 'cray':
                    self.__no_queues = 8
                    self.train_ert_min = 7200
                    self.train_ert_max = 86400
                    self.train_req_min = 32.0
                    self.train_req_max = 256.0
                    self.train_run_min = 1.0
                    self.train_run_max = 86000.0
                    self.train_wait_min = 1.0
                    self.train_wait_max = 86000.0
		
                self.dbscan_pred = DBSCAN_Based_SDScan(1, self.train_wait_max-self.train_wait_min)
		#print self.dbscan_pred
                self.job_histogram_list = {}
                self.saved_distr_vector = []
                self.time_since_last_save = 100

		
	#Calculate the RMS of a vector	(check if root mean square)
	def __calc_rms(self, vector):
				
		if len(vector) == 0:
			return 0
		else:
			rms = 0
			for k in range(len(vector)):
				rms += (vector[k]*vector[k])
			rms /= len(vector)
			rms = math.sqrt(rms)
			return rms			
	
	#Return the slab of the time
	def __getslab(self, time):
		if time <= 3600:
			return 0

		elif time <= 21600:
			return 1

		elif time <= 43200:
			return 2

		elif time <= 86400:
			return 3
		
		elif time <= 172800:
			return 4
		
		elif time <= 345600:
			return 5
		
		else:
			return 6
	
	def __round_time(self, time, factor):
        
		if time <= 3600: # 1 hour
			time = (int(time/900)+factor)*900 # 15 mins

		elif time <= 10800: # 3 hours
			time = (int(time/1800)+factor)*1800 # 30 mins

		elif time <= 21600: # 6 hours
			time = (int(time/3600)+factor)*3600 # 1 hour

		elif time <= 43200:# 12 hours
			time = (int(time/10800)+factor)*10800 # 3 hours
        
		elif time <= 86400: # 24 hours
			time = (int(time/21600)+factor)*21600 # 6 hours
        
		elif time <= 172800: # 2 days
			time = (int(time/43200)+factor)*43200 # 12 hours
			
		else:
			time = (int(time/86400)+factor)*86400 # 1 day
        
		if time < 0:
			time = 0    
            
		return time

	def __is_starving(self, job_id, starving_job_id):
		if self.__arr_time[job_id] - self.__arr_time[starving_job_id] > 86400:
			return True

	def __get_priority(self, priority_job):
		queue_id = self.__queue_id[priority_job]
		if queue_id == 0:
		    return 100
		elif queue_id == 1:
		    return 300
		elif queue_id == 2:
		    return 300
		elif queue_id == 3:
		    return 350
		elif queue_id == 4:
		    return 400
		else:
		    print "Not a CPU queue job_id and queue_id:",priority_job, queue_id
		

	# Calculate properties of the History Jobs
	def __set_job_properties(self, job_id, reqsize, ert):
                
                jobs_bf = 0 ; jobs_fcfs = 0 ; temp_prev_req_waitq = 0 ; temp_prev_ert_waitq = 0		
		running_elapsed_time = [] ; running_rem_time = [] ; waiting_elapsed_time = []
		
		
		for item in self.__waiting_job_id:
			if self.__req_size[item] <= reqsize and self.__est_runtime[item] <= ert and self.__queue_id[item] == self.__queue_id[job_id]:
				jobs_bf += 1	
			else:
				jobs_fcfs += 1				
		job_rank_combo = float(jobs_bf)/float(jobs_bf+jobs_fcfs)

		for item in self.__waiting_job_id:
			if self.__req_size[item] <= reqsize and self.__queue_id[item] == self.__queue_id[job_id]:
				temp_prev_req_waitq += self.__req_size[item]
				jobs_bf += 1	
			else:
				jobs_fcfs += 1				
		job_rank = float(jobs_bf)/float(jobs_bf+jobs_fcfs)
		
		jobs_bf = 0 ; jobs_fcfs = 0
		for item in self.__waiting_job_id:
			if self.__est_runtime[item] <= ert and self.__queue_id[item] == self.__queue_id[job_id]:
				temp_prev_ert_waitq += self.__est_runtime[item]
				temp_prev_req_waitq += self.__req_size[item]				
				jobs_bf += 1	
			else:
				jobs_fcfs += 1				
		job_rank_ert = float(jobs_bf)/float(jobs_bf+jobs_fcfs)
		
	
		proc_occ = self.__running_job_reqsize[:]
		self.__proc_occ_map[job_id] = proc_occ
		
		temp_prev_req_waitq += reqsize
		temp_prev_ert_waitq += ert
		
		self.__qsize_map[job_id] = jobs_bf+jobs_fcfs+1
		self.__job_rank_reqsize_map[job_id] = job_rank
		self.__job_rank_ert_map[job_id] = job_rank_ert
		self.__job_rank_combo_map[job_id] = job_rank_combo
		self.__free_node_map[job_id] = self.__free_nodes
		self.__proc_occ_map[job_id] = proc_occ
		self.__prev_req_waitq[job_id] = temp_prev_req_waitq
		self.__prev_ert_waitq[job_id] = temp_prev_ert_waitq
		
		self.__proc_occ_reqsize_rms[job_id] = self.__calc_rms(self.__running_job_reqsize)
		self.__proc_occ_ert_rms[job_id] = self.__calc_rms(self.__running_job_ert)
		if len(self.__running_job_reqsize) > 0:
			self.__proc_occ_reqsize_avg[job_id] = sum(self.__running_job_reqsize)/float(len(self.__running_job_reqsize))
			self.__proc_occ_ert_avg[job_id] = sum(self.__running_job_ert)/float(len(self.__running_job_ert))
			self.__proc_occ_cputime_avg[job_id] = sum(self.__running_job_cpu)/float(len(self.__running_job_cpu))
		else:
			self.__proc_occ_reqsize_avg[job_id] = 0
			self.__proc_occ_ert_avg[job_id] = 0
			self.__proc_occ_cputime_avg[job_id] = 0
			
		self.__queue_occ_reqsize_rms[job_id] = self.__calc_rms(self.__waiting_job_reqsize)
		self.__queue_occ_ert_rms[job_id] = self.__calc_rms(self.__waiting_job_ert)
		if len(self.__waiting_job_reqsize) > 0:
			self.__queue_occ_reqsize_avg[job_id] = sum(self.__waiting_job_reqsize)/float(len(self.__waiting_job_reqsize))
			self.__queue_occ_ert_avg[job_id] = sum(self.__waiting_job_ert)/float(len(self.__waiting_job_ert))
			self.__queue_occ_cputime_avg[job_id] = sum(self.__waiting_job_cpu)/float(len(self.__waiting_job_cpu))
		else:
			self.__queue_occ_reqsize_avg[job_id] = 0
			self.__queue_occ_ert_avg[job_id] = 0
			self.__queue_occ_cputime_avg[job_id] = 0
		
		#IBL Parameters begin
		temp_list_1 = [] 
                temp_list_2 = []
                temp_list_3 = []

		for k in range(self.__no_queues):
			temp_list_1.append(0)
			temp_list_2.append(0)
			temp_list_3.append(0)		
			
		for k in range(len(self.__running_job_id)):
			elapsed = self.__arr_time[job_id] - self.__start_time[self.__running_job_id[k]]
			remaining = self.__est_runtime[self.__running_job_id[k]] - elapsed
			running_elapsed_time.append(elapsed*self.__req_size[self.__running_job_id[k]])	
			running_rem_time.append(remaining*self.__req_size[self.__running_job_id[k]])
			#temp_list_1[self.__queue_id[job_id]-1] += 1
			#temp_list_2[self.__queue_id[job_id]-1] += (elapsed*self.__req_size[self.__running_job_id[k]])
			#temp_list_3[self.__queue_id[job_id]-1] += (remaining*self.__req_size[self.__running_job_id[k]])
			temp_list_1[0] += 1
			temp_list_2[0] += (elapsed*self.__req_size[self.__running_job_id[k]])
			temp_list_3[0] += (remaining*self.__req_size[self.__running_job_id[k]])

		self.__proc_occ_no_jobs[job_id] = temp_list_1 ### there is only 1 non zero element in temp_list_1 (we can as well consider it as sum(temp_list_1)
		self.__proc_occ_elapsed_time_total[job_id] = temp_list_2
		self.__proc_occ_remaining_time_total[job_id] = temp_list_3
		#print temp_list_1, temp_list_2, temp_list_3
		
		temp_list_1 = [] 
                temp_list_2 = []
                temp_list_3 = []
		for k in range(self.__no_queues):
			temp_list_1.append(0)
			temp_list_2.append(0)
			temp_list_3.append(0)
			
		for k in range(len(self.__waiting_job_id)):
			elapsed = self.__arr_time[job_id] - self.__arr_time[self.__waiting_job_id[k]]
			demand = self.__req_size[self.__waiting_job_id[k]]*self.__est_runtime[self.__waiting_job_id[k]]
			waiting_elapsed_time.append(elapsed*self.__req_size[self.__waiting_job_id[k]])	
			#temp_list_1[self.__queue_id[job_id]-1] += 1
			#temp_list_2[self.__queue_id[job_id]-1] += (elapsed*self.__req_size[self.__waiting_job_id[k]])
			#temp_list_3[self.__queue_id[job_id]-1] += demand
                        temp_list_1[0] += 1
			temp_list_2[0] += (elapsed*self.__req_size[self.__waiting_job_id[k]])
			temp_list_3[0] += demand

		self.__queue_occ_no_jobs[job_id] = temp_list_1
		self.__queue_occ_elapsed_time_total[job_id] = temp_list_2
		self.__queue_occ_demand_time_total[job_id] = temp_list_3
                
		q = self.__queue_group[job_id]
		self.user_queue_waiting_jobs[job_id] = 0.0
		self.user_queue_demand_wallclock_sum[job_id] = 0.0
		self.user_queue_demand_reqsize_sum[job_id] = 0.0
		self.user_queue_demand_ert_sum[job_id] = 0.0
                self.user_queue_elapsed_wallclock_sum[job_id] = 0.0

		self.record_waiting_jobs[job_id] = []

                self.sum_wait_ert[job_id] = 0.0
                self.sum_wait_req[job_id] = 0.0
                self.sum_wait_elapsed[job_id] = 0.0
                self.sum_run_ert[job_id] = 0.0
                self.sum_run_req[job_id] = 0.0
                self.sum_run_elapsed[job_id] = 0.0
		self.sum_higher_priority_jobs[job_id] = 0.0 	#cray_addition_sid
		self.sum_starving_jobs[job_id] = 0.0		#cray_addition_sid
		#self.queue_waiting_jobs[job_id] = 0.0
                
		if job_id == 53706:
			print "validation: waiting jobs: job_id=",job_id," from set job properties__running_jobs=",self.__waiting_job_id
			print "validation: running jobs: job_id=",job_id," from set job properties__running_jobs=",self.__running_job_id

		for wjob in self.__waiting_job_id:
                       

			if self.__queue_group[wjob] == q:
				self.record_waiting_jobs[job_id].append(wjob)
				#if job_id == 53706:
                                #	print "validation\t",wjob,"\trecord waiting jobs from set job properties\t",self.record_waiting_jobs
				if self.__get_priority(job_id) < self.__get_priority(wjob):
					self.sum_higher_priority_jobs[job_id] += 1
				if self.__is_starving(job_id, wjob):
					self.sum_starving_jobs[job_id] += 1
                                self.sum_wait_ert[job_id] += self.__est_runtime[wjob]
                                self.sum_wait_req[job_id] += self.__req_size[wjob]
                                self.sum_wait_elapsed[job_id] += self.__arr_time[job_id] - self.__arr_time[wjob]

				# populate user based queue features
				if self.__user_id[job_id] == self.__user_id[wjob]:
					self.user_queue_waiting_jobs[job_id] += 1
					self.user_queue_demand_wallclock_sum[job_id] += self.__req_size[wjob]*self.__est_runtime[wjob]
					self.user_queue_demand_reqsize_sum[job_id] += self.__req_size[wjob]
					self.user_queue_demand_ert_sum[job_id] += self.__est_runtime[wjob]
                                        self.user_queue_elapsed_wallclock_sum[job_id] += (self.__arr_time[job_id] - self.__arr_time[wjob])*self.__req_size[wjob]

		temp_user_queue_features = [self.user_queue_waiting_jobs[job_id], self.user_queue_demand_wallclock_sum[job_id], self.user_queue_demand_reqsize_sum[job_id], self.user_queue_demand_ert_sum[job_id], self.user_queue_elapsed_wallclock_sum[job_id]]
		if job_id == 53706:
			print "validation: TUQF: job_id=",job_id, " TUQF=", temp_user_queue_features

 
		for p in range(len(self.user_queue_features_min)):
			self.user_queue_features_min[p] = min(self.user_queue_features_min[p], temp_user_queue_features[p])
			self.user_queue_features_max[p] = max(self.user_queue_features_max[p], temp_user_queue_features[p])
			if job_id == 53706:
				print "validation: Feature_min[",p,"]: job_id=",job_id, " F_Min[",p,"]=", self.user_queue_features_min[p]
				print "validation: Feature_max[",p,"]: job_id=",job_id, " F_Max[",p,"]=", self.user_queue_features_max[p]
		if job_id == 53706:
			print "validation: FMIN: job_id=",job_id," FMIN=",self.user_queue_features_min
			print "validation: FMAX: job_id=",job_id," FMAX=",self.user_queue_features_max

                	print job_id, 'UQFM:', self.user_queue_features_max[0], self.user_queue_features_min[0]

                self.user_proc_running_jobs[job_id] = 0.0
		self.user_proc_demand_wallclock_sum[job_id] = 0.0
		self.user_proc_demand_reqsize_sum[job_id] = 0.0
		self.user_proc_demand_ert_sum[job_id] = 0.0
                self.user_proc_elapsed_wallclock_sum[job_id] = 0.0

		self.record_running_jobs[job_id] = []
                #print job_id,"\t","rr from top",self.__running_job_id	
		for rjob in self.__running_job_id:
                       

			if self.__queue_group[rjob] == q:
                                #kruthika
                                #print rjob,"q from rjob\t",q
				
				self.record_running_jobs[job_id].append(rjob)
				if job_id == 53706:
					print "validation\t",wjob,"\trecord running jobs from set job properties\t",self.record_running_jobs
                                #print rjob," est from rjob\t",self.__est_runtime[rjob]
                                self.sum_run_ert[job_id] += self.__est_runtime[rjob]
                                self.sum_run_req[job_id] += self.__req_size[rjob]
                                self.sum_run_elapsed[job_id] += self.__arr_time[job_id] - (self.__arr_time[rjob] + self.__wait_time[rjob])

                                if self.__user_id[job_id] == self.__user_id[rjob]:
                                    #print job_id,"job_id user id\t",self.__user_id[job_id]
				    #print rjob,"rjob user id\t",self.__user_id[rjob]
                                    self.user_proc_running_jobs[job_id] += 1
                                    self.user_proc_demand_wallclock_sum[job_id] += self.__req_size[rjob]*self.__est_runtime[rjob]
                                    self.user_proc_demand_reqsize_sum[job_id] += self.__req_size[rjob]
                                    #print rjob," est from rjob in user id\t",self.__est_runtime[rjob]
                                    self.user_proc_demand_ert_sum[job_id] += self.__est_runtime[rjob]
                                    self.user_proc_elapsed_wallclock_sum[job_id] += (self.__arr_time[job_id]-self.__start_time[rjob])*self.__req_size[rjob]
                temp_user_proc_features = [self.user_proc_running_jobs[job_id], self.user_proc_demand_wallclock_sum[job_id], self.user_proc_demand_reqsize_sum[job_id], self.user_proc_demand_ert_sum[job_id], self.user_proc_elapsed_wallclock_sum[job_id]]
		if job_id == 53706:
			print "validation: TUPF: job_id=",job_id, " TUPF=", temp_user_proc_features
		for p in range(len(self.user_proc_features_min)):
			self.user_proc_features_min[p] = min(self.user_proc_features_min[p], temp_user_proc_features[p])
			self.user_proc_features_max[p] = max(self.user_proc_features_max[p], temp_user_proc_features[p])
		if job_id == 53706:
			print "validation: PMIN: job_id=",job_id," PMIN=",self.user_proc_features_min
			print "validation: PMAX: job_id=",job_id," PMAX=",self.user_proc_features_max

                sum_list = [self.sum_wait_ert[job_id], self.sum_wait_req[job_id], self.sum_wait_elapsed[job_id], self.sum_run_ert[job_id], self.sum_run_req[job_id], self.sum_run_elapsed[job_id], self.sum_higher_priority_jobs[job_id], self.sum_starving_jobs[job_id]]

                for p in range(len(sum_list)):
                    self.sum_test_min[p] = min(self.sum_test_min[p], sum_list[p])
                    self.sum_test_max[p] = max(self.sum_test_max[p], sum_list[p])
                    #print job_id,"sum list\t",sum_list[p]
                    #print job_id,"max\t",self.sum_test_max[p]
                self.job_feature_vector[job_id] = [self.__req_size[job_id],
                                                   self.__est_runtime[job_id],
                                                   0.0,
                                                   0.0,

                                                   self.user_queue_waiting_jobs[job_id],
                                                   self.user_queue_demand_wallclock_sum[job_id],
                                                   self.user_queue_demand_reqsize_sum[job_id],
                                                   self.user_queue_demand_ert_sum[job_id],

                                                   self.user_proc_running_jobs[job_id],
                                                   self.user_proc_demand_wallclock_sum[job_id],
                                                   self.user_proc_demand_reqsize_sum[job_id],
                                                   self.user_proc_demand_ert_sum[job_id],

                                                   self.sum_wait_ert[job_id], 
                                                   self.sum_wait_req[job_id], 
                                                   self.sum_wait_elapsed[job_id], 

                                                   self.sum_run_ert[job_id], 
                                                   self.sum_run_req[job_id], 
                                                   self.sum_run_elapsed[job_id],

						   self.sum_higher_priority_jobs[job_id],
						   self.sum_starving_jobs[job_id]]
		if job_id == 53706:
			print "validation\t",job_id,"\tjob feature vector\t",self.job_feature_vector[job_id]
                #print job_id,"Sum of run ert from job vecto\t",self.sum_run_ert[job_id]
                if job_id in check_list:
                    #print 'For job', job_id//
                    with open('n.txt','a') as f2:	
                                f2.write("{0}\n".format("For job"))
                    for wjob in self.__waiting_job_id:
			if self.__queue_id[wjob] == q and self.__req_size[wjob] >= self.__req_size[job_id]:
                           #print wjob, 'req=',self.__req_size[wjob], 'ert=', self.__est_runtime[wjob], 'elapsed=',self.__arr_time[job_id] - self.__arr_time[wjob]//
                           with open('n.txt','a') as f2:	
                                f2.write("{0}\n".format("list features")) 
                if self.ENABLE_DISTRIBUTIONS:
			TJW_ert = [0]; TJW_req_size = [0]; TJW_elapsed = [0]; TJR_ert = [0]; TJR_req_size = [0]; TJR_elapsed = [0];
			TJW_cpu = [0]; TJR_cpu = [0];
                        DCount = sum(self.selected_distribution_list)
			#waiting_job_ids = self.recordistr_list_initiald_waiting_jobs[job_id]
                        waiting_job_ids = self.record_waiting_jobs[job_id]
			running_job_ids = self.record_running_jobs[job_id]
			TJ_wait_count = len(waiting_job_ids)
			TJ_run_count = len(running_job_ids)
			target_job_distribution_bin_values = []

			for item in waiting_job_ids:
				if self.selected_distribution_list[0]:
					TJW_ert.append(self.__est_runtime[item])
				if self.selected_distribution_list[1]:
					TJW_req_size.append(self.__req_size[item])
				if self.selected_distribution_list[2]:
					TJW_elapsed.append(self.__arr_time[job_id] - self.__arr_time[item])
				if self.selected_distribution_list[3]:
					TJW_cpu.append(self.__est_runtime[item]*self.__req_size[item])
				
			for item in running_job_ids:
				if self.selected_distribution_list[4]:
					TJR_ert.append(self.__est_runtime[item])
				if self.selected_distribution_list[5]:
					TJR_req_size.append(self.__req_size[item])
				if self.selected_distribution_list[6]:
					TJR_elapsed.append(self.__arr_time[job_id] - (self.__start_time[item] + self.__wait_time[item]))
				if self.selected_distribution_list[7]:
					TJR_cpu.append(self.__est_runtime[item]*self.__req_size[item])
                        #print TJR_req_size
                        # Compute histograms for each distribution
			W = 0
			R = 1		
			distr_list_initial = [TJW_ert, 
                                              TJW_req_size, 
                                              TJW_elapsed, 
                                              TJW_cpu, 
                                              TJR_ert, 
                                              TJR_req_size, 
                                              TJR_elapsed,
                                              TJR_cpu]
                        distr_list_min = [self.train_ert_min, 
                                          self.train_req_min, 
                                          self.train_wait_min, 
                                          self.train_ert_min*self.train_req_min,
                                          self.train_ert_min, 
                                          self.train_req_min, 
                                          self.train_run_min,
                                          self.train_ert_min*self.train_req_min]

                        distr_list_max = [self.train_ert_max, 
                                          self.train_req_max, 
                                          self.train_wait_max,
                                          self.train_ert_max*self.train_req_max,
                                          self.train_ert_max, 
                                          self.train_req_max,
                                          self.train_run_max,
                                          self.train_ert_max*self.train_req_max]
                        
                        histogram_type_initial = [W,W,W,W,R,R,R,R]
			distr_list = []  #all histogram data
			histogram_type = []
                        dl_min = []
                        dl_max = []
                        #print 'Picking',
			for i in range(len(self.selected_distribution_list)):
				if self.selected_distribution_list[i]:
                                        #print i," ",
					distr_list.append((distr_list_initial[i], distr_list_min[i], distr_list_max[i]))
					histogram_type.append(histogram_type_initial[i])
                        #print ""
			histogram_list = [numpy.histogram(distribution[0], bins = self.bin_count, range=(distribution[1], distribution[2])) for distribution in distr_list]       #has all the histograms
                        final_histogram_list = []
                        #print job_id
                        #print ""
                        for i in range(DCount):
                            hist = histogram_list[i]    #select one histogram
                            hist_type = histogram_type[i]
                            adjusted_bin_count = hist[0]     #bin count of selected histogram
                            if hist_type == 0:
                                if TJ_wait_count != 0:
                                    adjusted_bin_count = adjusted_bin_count/float(TJ_wait_count)   #modified bin count(/ by TJ_wait_count)
                            else:
                                if TJ_run_count != 0:
                                    adjusted_bin_count = adjusted_bin_count/float(TJ_run_count)	   #modified bin count(/ by TJ_run_count)
                            final_histogram_list.append(adjusted_bin_count)
                            if 0 and i >= 3:
                                #print job_id, adjusted_bin_count//
                                with open('n.txt','a') as f2:	
                                 f2.write("{0}\n".format("adjustcount"))
                            #print i,adjusted_bin_count
                        #print job_id, final_histogram_list
                        
                        self.job_histogram_list[job_id] = final_histogram_list#modified(normalized) bin count of each histogram of job_id 

	def chi_square(self, P, Q):
		num = 0.0
		den = 0.0
		chi2 = 0.0
                
		for pos in range(len(P)):
                        
			num = (P[pos] - Q[pos])*(P[pos] - Q[pos])*1.0
			den = (P[pos] + Q[pos])*1.0
			if den!=0:
				chi2 += num/den
		chi2 = chi2
		#if chi2 <= 0:
		#	#print P, Q
		return chi2

        def weighted_chi_square(self, P, Q, W, zero_count):
                if len(W) == zero_count:
                    return 0
		num = 0.0
		den = 0.0
		chi2 = 0.0
		for pos in range(len(P)):
			num = W[pos]*(P[pos] - Q[pos])*(P[pos] - Q[pos])*1.0
			den = (P[pos] + Q[pos])*1.0
			if den!=0:
				chi2 += num/den
		chi2 = 1.0*chi2
		return chi2
        
        def compute_euclidean_norm(self, X):
            value = 0.0
            for i in range(len(X)):
                value += X[i]*X[i]
            return math.sqrt(value)

        def get_correlation_vector_local(self, job_id):
            # find the history job S with minimum norm - to get an origin point
            # to get S, get all distribution data, feature data etc. and compute euclidean norm and get minimum
            # Currently using only distribution data to get correlations for chi-square (bin unweighted)
            # there is an issue with normalization to use other features also
            
            # a simple caching mechanism to speed up the correlation calculation
            if self.time_since_last_save > 100:
                self.saved_distr_vector = []
                self.time_since_last_save = 0

            if len(self.saved_distr_vector) != 0:
                self.time_since_last_save += 1
                return self.saved_distr_vector

            
            DCount = sum(self.selected_distribution_list)
            all_correlations = []
            for DINDEX in range(DCount):                
                norm_list = []
                norm_min = float('inf')
                min_job_id = 0
                wait_time_list = []
                for j in self.__history_jobs:
                    if self.__arr_time[job_id] >= (self.__arr_time[j] + self.__wait_time[j]):
                        temp_vector = self.job_histogram_list[j][DINDEX].tolist()  #job_histogram_list is dictionary having values as normalized bin count for each histogram. tolist() converts array returned by numpy.histogram[0] to list
                        norm_value = self.compute_euclidean_norm(temp_vector)
                        norm_list.append(norm_value)	#contains euclidian norm of 'DINDEX' dist of all jobs
                        wait_time_list.append(self.__wait_time[j])
               
                correlation_list = 0.0
                
                X = norm_list
                Y = wait_time_list
                
                #if dindex == 3:
                #    print X[:100]
                #    print Y[:100]
                #plt.scatter(X,Y)
                #plt.show()
            
                correlation_value = spearmanr(X, Y)
                #print correlation_value

                if len(X) == 0 or math.isnan(correlation_value[0]) or abs(correlation_value[0]) < 0.1:
                    correlation_list = 0.0
                else:
                    correlation_list = abs(correlation_value[0])
                all_correlations.append(correlation_list)
            if sum(all_correlations) == 0.0:
                all_correlations = [1.0]*len(all_correlations)
            self.saved_distr_vector = all_correlations

            return all_correlations

        def get_correlation_vector(self, job_id):
            # find the history job S with minimum norm - to get an origin point
            # to get S, get all distribution data, feature data etc. and compute euclidean norm and get minimum
            # Currently using only distribution data to get correlations for chi-square (bin unweighted)
            # there is an issue with normalization to use other features also
            
            # a simple caching mechanism to speed up the correlation calculation
            if self.time_since_last_save > 100:	#if greater than 100 then recompute
                self.saved_distr_vector = []
                self.time_since_last_save = 0

            if len(self.saved_distr_vector) != 0:	#if less than 100 return saved if not reset in about section
                self.time_since_last_save += 1
                return self.saved_distr_vector
            
            DCount = sum(self.selected_distribution_list)
            norm_list = []
            norm_min = float('inf')
            min_job_id = 0
            for j in self.__history_jobs:
                if self.__arr_time[job_id] >= (self.__arr_time[j] + self.__wait_time[j]):
                    temp_vector = []
                    for dindex in range(DCount):
                        temp_vector.extend(self.job_histogram_list[j][dindex].tolist()) #it is dictionary
                    norm_value = self.compute_euclidean_norm(temp_vector)
                    norm_list.append([norm_value, j])
                    if norm_value < norm_min:
                        norm_min = norm_value
                        min_job_id = j
            #print norm_min, min_job_id
            max_dist_list = [-float('inf')]*DCount
            distribution_distance_list = [[] for i in range(DCount)]
            wait_time_list = []
            for j in self.__history_jobs:
                if self.__arr_time[job_id] >= (self.__arr_time[j] + self.__wait_time[j]):
                    temp_dist_vector = []
                    for dindex in range(DCount):
                        distance_value = self.chi_square(self.job_histogram_list[min_job_id][dindex], self.job_histogram_list[j][dindex])
                        temp_dist_vector.append(distance_value)
                        distribution_distance_list[dindex].append(distance_value)
                    wait_time_list.append(self.__wait_time[j])
            correlation_list = []
            for k in range(DCount):
                X = distribution_distance_list[k]
                Y = wait_time_list
                correlation_value = spearmanr(X, Y)
                #print correlation_value
                if len(X) == 0 or math.isnan(correlation_value[0]) or abs(correlation_value[0]) < 0.1:
                    correlation_list.append(0.0)
                else:
                    correlation_list.append(abs(correlation_value[0]))
            #print correlation_list
            
            self.saved_distr_vector = correlation_list

            return correlation_list

        def compute_list_distance(self, my_wait_erts, hwait_erts, is_log):

            if len(my_wait_erts) == 0 and len(hwait_erts) == 0:
                return 0
            min_ert = 0
            max_ert = 0
            if len(my_wait_erts) != 1 and len(hwait_erts) != 0:
                min_ert = min(min(my_wait_erts), min(hwait_erts))
                max_ert = max(max(my_wait_erts), max(hwait_erts))
            else:
                if len(my_wait_erts) == 0:
                    min_ert = min(hwait_erts)
                    max_ert = max(hwait_erts)
                else:
                    min_ert = min(my_wait_erts)
                    max_ert = max(my_wait_erts)

            bin_count = 10
            base_val = 5.0
            x_width = float(max_ert - min_ert)*(base_val-1.0)/(math.pow(base_val,bin_count)-1.0)

            ## ERT binning ##
            # set up the bins and freq counters
            ert_bin_start = [0]*bin_count
            my_ert_freq = [0]*bin_count
            hert_freq = [0]*bin_count
            ert_bin_width = float(max_ert - min_ert)/bin_count
            current_pos = min_ert
            for i in range(bin_count):
                ert_bin_start[i] = current_pos
                if is_log == 1:
                    current_pos += x_width*math.pow(base_val,i)
                else:
                    current_pos += ert_bin_width

            # populate the freq counters
            for item in my_wait_erts:
                for pos in range(bin_count):
                    if ert_bin_start[pos] <= item:
                        if pos+1 == bin_count:
                            my_ert_freq[pos] += 1
                        elif item <= ert_bin_start[pos+1]:
                            my_ert_freq[pos] += 1
                        else:
                            continue
            if len(my_wait_erts) != 0:
                for mef_index in range(len(my_ert_freq)):
                    #if is_log:
                    #    my_ert_freq[mef_index] = my_ert_freq[mef_index]/(float(len(my_wait_erts))*x_width*math.pow(base_val,mef_index))
                    #else:
                    my_ert_freq[mef_index] = my_ert_freq[mef_index]/(float(len(my_wait_erts)))

            for item in hwait_erts:
                for pos in range(bin_count):
                    if ert_bin_start[pos] <= item:
                        if pos+1 == bin_count:
                            hert_freq[pos] += 1
                        elif item <= ert_bin_start[pos+1]:
                            hert_freq[pos] += 1
                        else:
                            continue

            if len(hwait_erts) != 0:
                for hef_index in range(len(hert_freq)):
                    #if is_log:
                    #    hert_freq[hef_index] = hert_freq[hef_index]/(float(len(hwait_erts))*x_width*math.pow(base_val,mef_index))
                    #else:
                    hert_freq[hef_index] = hert_freq[hef_index]/(float(len(hwait_erts)))

            wait_ert_distance = 0.0
            wait_ert_distance = self.chi_square(my_ert_freq, hert_freq)
            return wait_ert_distance

        def compute_distribution_distances(self, job_id):

            hq = q = self.__queue_id[job_id]
            #print "hq",hq
            my_wait_erts = []
            my_wait_req_sizes = []
            my_wait_till_now = []
            my_run_erts = []
            my_run_req_sizes = []
            my_run_till_now = []

            waiting_job_ids = self.record_waiting_jobs[job_id]
            #print waiting_job_ids
            for item in waiting_job_ids:
                my_wait_erts.append(self.__est_runtime[item])
                #print item,"\t",my_wait_erts
                my_wait_req_sizes.append(self.__req_size[item])
                #print item,"\t",my_wait_req_sizes
                my_wait_till_now.append(self.__arr_time[job_id] - self.__arr_time[item])
                #print item,"\t",my_wait_till_now

            running_job_ids = self.record_running_jobs[job_id]
            #print  job_id,"\t","rr from dd",running_job_ids
            for item in running_job_ids:
                my_run_erts.append(self.__est_runtime[item])
                #print item,"\t",my_run_erts
                my_run_req_sizes.append(self.__req_size[item])
                #print item,"\t",my_run_req_sizes
                my_run_till_now.append(self.__arr_time[job_id] - self.__start_time[item])
                #print item,"\t",my_run_till_now
            current_waiting_count = len(my_wait_erts)
            current_running_count = len(my_run_erts)

            distances = []
            heom_values = []
            max_dist = 0.0
            min_rs = self.__req_size[job_id]
            max_rs = self.__req_size[job_id]
            min_ert = self.__est_runtime[job_id]
            max_ert = self.__est_runtime[job_id]
            min_time = self.__arr_time[job_id]
            max_time = self.__arr_time[job_id]

            time_dist_list = []
            DCount = sum(self.selected_distribution_list)
            #print "DCount",DCount
            max_dist_list = [-float('inf')]*DCount
            #print "max_dist_list",max_dist_list
            distribution_distance = {}
            #print job_id,"\t",self.__history_jobs
            for j in self.__history_jobs:	
                if self.__arr_time[job_id] >= (self.__arr_time[j] + self.__wait_time[j]):
                    if self.__queue_id[job_id] != self.__queue_id[j]:
                        continue
                    hjob_id = j
                    hwait_erts = []
                    hwait_req_sizes = []
                    hwait_till_now = []
                    hrun_erts = []
                    hrun_req_sizes = []
                    hrun_till_now = []

                    hwaiting_job_ids = self.record_waiting_jobs[hjob_id]
                    #print hjob_id,"\t",hwaiting_job_ids
                    for item in hwaiting_job_ids:
                        hwait_erts.append(self.__est_runtime[item])
                        #print item,"\t",hwait_erts
                        hwait_req_sizes.append(self.__req_size[item])
                        #print item,"\t",hwait_req_sizes
                        hwait_till_now.append(self.__arr_time[hjob_id] - self.__arr_time[item])
                        #print item,"\t",hwait_till_now
                    #print "jobid",j,"hwait_erts",hwait_erts
                    hrun_job_ids = self.record_running_jobs[hjob_id]
                    #print hjob_id,"\t",hrun_job_ids
                    #print self.record_running_jobs
                    for item in hrun_job_ids:
                        hrun_erts.append(self.__est_runtime[item])
                        #print item,"\t",hrun_erts
                        hrun_req_sizes.append(self.__req_size[item])
                        #print item,"\t",hrun_req_sizes
                        hrun_till_now.append(self.__arr_time[hjob_id] - self.__start_time[item])
                        #print item,"\t",hrun_till_now
                    dwreq = self.compute_list_distance(my_wait_req_sizes, hwait_req_sizes, 0)
                    dwert = self.compute_list_distance(my_wait_erts, hwait_erts, 0)
                    dwtill = self.compute_list_distance(my_wait_till_now, hwait_till_now, 0)

                    drreq = self.compute_list_distance(my_run_req_sizes, hrun_req_sizes, 0)
                    drert = self.compute_list_distance(my_run_erts, hrun_erts, 0)
                    drtill = self.compute_list_distance(my_run_till_now, hrun_till_now, 0)
                    temp_dist_vector = [dwreq, dwert, dwtill, drreq, drert, drtill]
                    #print "temp_dist_vector", temp_dist_vector
                    for i in range(6):
                        max_dist_list[i] = max(max_dist_list[i], temp_dist_vector[i])
                        #print "max_dist", max_dist_list[i]
                    distribution_distance[hjob_id] = temp_dist_vector
            #print distribution_distance.keys()
            for key in distribution_distance.keys():
                for dindex in range(DCount):
                    if max_dist_list[dindex]:
                        distribution_distance[key][dindex] = 1.0*distribution_distance[key][dindex]/max_dist_list[dindex]
                        #print distribution_distance[key][dindex]
            #print distribution_distance['13221']
            return distribution_distance
        def get_pred_from_unweighted_distribution(self, job_id, distribution_distance):
            individual_distance_map = {}
            individual_wait_times = {}
            weight_list = [1.0]*8
            for j in self.__history_jobs:	
                    if self.__arr_time[job_id] >= (self.__arr_time[j] + self.__wait_time[j]):
                            if self.__queue_id[job_id] != self.__queue_id[j]:
                                    continue
                            distance = []
                            ### Job features ###
                            if self.__req_size[job_id] == self.__req_size[j]:
                                    distance.append(0)
                            else:
                                    distance.append(1)
                            ert_dist = float(abs(self.__est_runtime[job_id] - self.__est_runtime[j]))/float(self.__ert_max - self.__ert_min)
                            distance.append(ert_dist)
                            if self.ENABLE_DISTRIBUTIONS: 
                                distance.extend(distribution_distance[j])
                            individual_distance_map[j] = distance
                            #print job_id,"indiv dis from unweigth\t",individual_distance_map[j]

                            individual_wait_times[j] = self.__wait_time[j]
        
            if len(individual_distance_map) == 0:
                return False
	    #printValidation
	    #print "Validation: model: job_id=",job_id," pred 1_Web"
            dbscan_pred_output, pred_list, dmap, invalid_output = self.dbscan_pred.get_prediction_plain(individual_distance_map, individual_wait_times, job_id, weight_list,self.__wait_time[job_id])
            if not invalid_output:
                #print job_id, 'actual=', self.__wait_time[job_id], 'pred=', dbscan_pred_output, 'run=', self.__real_run_time[job_id]
                self.PredictedWaitingTime[job_id] = dbscan_pred_output
                #print "Prediction 3"
                return False
            else:
                return True
        def get_pred_from_weighted_distribution(self, job_id, distribution_distance, weight_array):
            #print "Hi from get pred from weighted distribution"
            individual_distance_map = {}
            individual_wait_times = {}
            weight_list = [1.0, 1.0]
            weight_list.extend(weight_array)
            for j in self.__history_jobs:	
                    if self.__arr_time[job_id] >= (self.__arr_time[j] + self.__wait_time[j]):
                            if self.__queue_id[job_id] != self.__queue_id[j]:
                                    continue
                            distance = []
                            ### Job features ###
                            if self.__req_size[job_id] == self.__req_size[j]:
                                    distance.append(0)
                            else:
                                    distance.append(1)
                            ert_dist = float(abs(self.__est_runtime[job_id] - self.__est_runtime[j]))/float(self.__ert_max - self.__ert_min)
                            distance.append(ert_dist)
                            if self.ENABLE_DISTRIBUTIONS: 
                                distance.extend(distribution_distance[j])
                            individual_distance_map[j] = distance
                            individual_wait_times[j] = self.__wait_time[j]
        
            if len(individual_distance_map) == 0:
                return False, 0
	    #printValidation
	    #print "Validation: model: job_id=",job_id," pred 2_Web"
            dbscan_pred_output, pred_list, dmap, invalid_output = self.dbscan_pred.get_prediction_plain(individual_distance_map, individual_wait_times, job_id, weight_list,self.__wait_time[job_id])#get_prediction_plain of line no 333. pred_list is a list of tuple of (overall dist, current wait, and job id) for all jobs
            if not invalid_output:
                #print job_id, 'actual=', self.__wait_time[job_id], 'pred=', dbscan_pred_output, 'run=', self.__real_run_time[job_id]
                #NOT IN PRAKASH (Kruthika)
                #print "Prediction 1"
                self.PredictedWaitingTime[job_id] = dbscan_pred_output
                return False, pred_list
            else:
                return True, pred_list
	def __get_qwt_strategy_based(self, job_id, wait_time):      #wait_time is wait time of job_id
		predicted_wait = 0 ; 
                
                ## for test ##
                '''
                if self.__queue_id[job_id] == 3:
                    print job_id,
                self.__ibl_predicted_wait_time[job_id] = predicted_wait
                return predicted_wait
                '''
                ## begin prediction ##
		prediction_list = [] ; distance_map = {}
		##
		# DISTR: Compute distributions and their bin edges for current job
		##
		distribution_distance = {}
		distribution_summary = {}
                distribution_bin_values = []
                distribution_bin_values_map = {}
                distribution_waits = []
                history_count = 0
		DCount = sum(self.selected_distribution_list)  #line no 783
 		max_dist_list = [-float('inf')]*DCount	#multiplication will result in max_dist_list having Dcount element all -inf

		individual_distance_map = {}
		individual_wait_times = {}
                print job_id, 'Queue:', self.queue_id_map[self.__queue_id[job_id]]
                distr_correlation_output = []
                history_job_id_list=[]
                #kruthika
                #print "dis",self.ENABLE_DISTRIBUTIONS
                #print "feaurre",self.ENABLE_SUMMARY_FEATURES
		## Calculate distribution distances for all relevant jobs -- begin
                if self.ENABLE_DISTRIBUTIONS:
                    # We need to fill up the distribution_distance array with normalized distance value for each of the 6 distributions.
                    # Find histogram of the current job and add extra bins at the left and right end.
                    # Find histogram for each past job using the bin limits derived for the current job
                    # Compute distance for each histogram of each past job
                    # Normalize the distances using the maximum and fill up the array
                    #print "hi"
                    # Unweighted case
                    distr_correlation_output = [1.0]*DCount
                    if test_use_weights_for_distributions:
                        #print "Hi I am inside the test use weights for distributions"
                        distr_correlation_output = self.get_correlation_vector_local(job_id)  #returns weight of six distributions
                    distribution_distance = self.compute_distribution_distances(job_id)       #returns distribution_distance[job][six distributions] in range [0,1]
                    #if job_id == 13221:  
                      #print distribution_distance
                    #kruthika
                    #for j in self.__history_jobs:
                        
                        #print "Job ID:",j,"\t","Distribution distance:",distribution_distance[j],"\n"
   
                    #Equal values are assigned a rank that is the average of the ranks that
                  
                    
                    # USE UNWEIGHTED DISTRIBUTION TO SEE IF OUTLIERS ARE THERE...IF THEY ARE THERE WE'LL DO RIDGE
                    do_regression, pred_list = self.get_pred_from_weighted_distribution(job_id, distribution_distance, distr_correlation_output)
                    print 'Do_regression for:',"\t",job_id,"\t", do_regression
                    self.__ibl_predicted_wait_time[job_id] = 0  

                    if  0 and job_id in check_list and pred_list != 0:
                        '''
                        print ""
                        print "----------------"
                        print pred_list[:20]
                        print "----------------"
                        print ""
                        for item in pred_list[:20]:
                            item_id = item[2]
                            print item_id, item[1], self.job_feature_vector[item_id]

                        print job_id, 'actual=', self.__wait_time[job_id], 'pred=', predicted_wait, 'run=', self.__real_run_time[job_id]
                        print 'R:', self.__req_size[job_id], 'ERT:', self.__est_runtime[job_id], 'U:', self.__user_id[job_id], 'E:', self.__exec_id[job_id], 'G:', self.__grp_id[job_id], 'Run:', self.__real_run_time[job_id], 'Q:', self.__queue_id[job_id]
                        #print 'WA_Pred=', wa_predicted_wait
                        '''
                        pX = []   #will have distance
                        pY = []   #will have time

                        x_percent = 1.0
                        top_x_percent = int(x_percent*len(pred_list))
                        count_x_percent = 0



                        while count_x_percent < top_x_percent:
                            item = pred_list[count_x_percent]
                            pX.append(item[0])
                            pY.append(item[1] - self.__wait_time[job_id])
                            count_x_percent += 1



                        #print len(pX), len(pY)
                        #print pX[:100], pY[:100]

                        plt.clf()
                        plt.scatter(pX,pY)
                        #title = 'Job id: ' + str(job_id) + ' Wait time= ' + str(self.__wait_time[job_id]) + ' D= ' + prediction_str # + ' Pred=' + str(predicted_wait)
                        title = str(job_id)
                        plt.title(title)
                        plt.xlabel('distance')
                        plt.ylabel('var. wait time')
                        #plt.show()
                        name = 'TyroneDistr' + str(job_id) + '.png'
                        #plt.show()
                        plt.savefig(name)
                  
                    if not do_regression:
                        return 0
                    
                    # ELSE WE'LL DO DBSCAN AND GET THE PREDICTION

                    #print "", distribution_distance[key], ""
                #print distribution_distance
		## 
		# Get correlation based weights
		# Get a set C of k jobs which are closest wrt req size and ert
		# Using C, compute correlations for each feature wrt wait time
		# Use the correlations as weights
		## 
                
		## Compute individual features for top_k jobs 
                #print job_id, "job feature \t",self.job_feature_vector[job_id]
                #print job_id, 'Required prediction=', self.__wait_time[job_id]
		correlation_features = []
		correlation_targets = []
                X_REG = []
                Y_REG = []
                
                my_his = []
                for j in self.__history_jobs:	
                        if self.__arr_time[job_id] >= (self.__arr_time[j] + self.__wait_time[j]):
                                my_his.append(j)
                                if self.__queue_id[job_id] != self.__queue_id[j]:
                                        continue
                                h_id = j
                                my_his.append(h_id)
                                this_feature_list = []
                                ''' 
                                # Test Li features 
                                this_feature_list.append(self.__grp_id[h_id])
                                this_feature_list.append(self.__user_id[h_id])
                                this_feature_list.append(self.__queue_id[h_id])
                                this_feature_list.append(self.__queue_id[h_id])

                                this_feature_list.append(self.__proc_occ_no_jobs[h_id][0])
                                this_feature_list.append(self.__queue_occ_no_jobs[h_id][0])
                                this_feature_list.append(self.__proc_occ_elapsed_time_total[h_id][0])
                                this_feature_list.append(self.__proc_occ_remaining_time_total[h_id][0])
                                this_feature_list.append(self.__queue_occ_elapsed_time_total[h_id][0])
                                this_feature_list.append(self.__queue_occ_demand_time_total[h_id][0])

                                this_feature_list.append(self.__grp_id[h_id])
                                this_feature_list.append(self.__user_id[h_id])
                                this_feature_list.append(self.__exec_id[h_id])
                                this_feature_list.append(self.__req_size[h_id])
                                this_feature_list.append(self.__est_runtime[h_id])
                                this_feature_list.append(self.__arr_time[h_id])

                                this_feature_list.append(self.user_queue_waiting_jobs[h_id])
                                this_feature_list.append(self.user_queue_demand_wallclock_sum[h_id])
                                this_feature_list.append(self.user_queue_demand_reqsize_sum[h_id])
                                this_feature_list.append(self.user_queue_demand_ert_sum[h_id])


                                ### Queue and processor features ###
                                this_feature_list.append(self.__free_node_map[h_id])
                                this_feature_list.append(self.__proc_occ_no_jobs[h_id][0])
                                this_feature_list.append(self.__queue_occ_no_jobs[h_id][0])
                                this_feature_list.append(self.__proc_occ_elapsed_time_total[h_id][0])
                                this_feature_list.append(self.__proc_occ_remaining_time_total[h_id][0])
                                this_feature_list.append(self.__queue_occ_elapsed_time_total[h_id][0])
                                this_feature_list.append(self.__queue_occ_demand_time_total[h_id][0])
                                this_feature_list.append(self.__queue_occ_demand_time_total[h_id][0])
                                '''

                                ### Summaries of distribution based features ###
                                #this_feature_list += distribution_summary[h_id]

                                ### User based features ###

                                #this_feature_list.extend(distribution_summary[h_id])
                                this_feature_list.append(self.__req_size[h_id])
                                
                                this_feature_list.append(self.__est_runtime[h_id])
                                this_feature_list.append(0.0)#self.__arr_time[h_id])
                                this_feature_list.append(0.0)#-self.__free_node_map[h_id])
                                
                                feature_id = 0
                                feature_name = self.user_queue_waiting_jobs
                                #if h_id == 5000:
                                  # print h_id,"feature name\t",feature_name
                                UQ_waiting_job_count = float(abs(feature_name[h_id]))
                                feature_id = 1
                                feature_name = self.user_queue_demand_wallclock_sum
                                UQ_demand_wallclock = float(abs(feature_name[h_id]))
                                feature_id = 2
                                feature_name = self.user_queue_demand_reqsize_sum
                                UQ_demand_reqsize = float(abs(feature_name[h_id]))
                                feature_id = 3
                                feature_name = self.user_queue_demand_ert_sum
                                UQ_demand_ert = float(abs(feature_name[h_id]))

                                #UQ_elapsed_wallclock = self.user_queue_elapsed_wallclock_sum[h_id]
                                
                                this_feature_list += [UQ_waiting_job_count, 
                                                      UQ_demand_wallclock, 
                                                      UQ_demand_reqsize, 
                                                      UQ_demand_ert]#, UQ_elapsed_wallclock]
                                this_feature_list += [self.user_proc_running_jobs[h_id]]
                                this_feature_list += [self.user_proc_demand_wallclock_sum[h_id]]
                                this_feature_list += [self.user_proc_demand_reqsize_sum[h_id]]
                                this_feature_list += [self.user_proc_demand_ert_sum[h_id]]
                                
                                '''
                                this_feature_list.append(self.__proc_occ_no_jobs[h_id][0])
                                this_feature_list.append(self.__queue_occ_no_jobs[h_id][0])
                                this_feature_list.append(self.__proc_occ_elapsed_time_total[h_id][0])
                                this_feature_list.append(self.__proc_occ_remaining_time_total[h_id][0])
                                this_feature_list.append(self.__queue_occ_elapsed_time_total[h_id][0])
                                this_feature_list.append(self.__queue_occ_demand_time_total[h_id][0])
                                this_feature_list.append(self.__queue_occ_demand_time_total[h_id][0])
                                ###
                                ## fix the below part when its used
                                ###

                                distance.append(float(abs(self.sum_wait_ert[job_id]-self.sum_wait_ert[j])/float(self.sum_test_max[0]-self.sum_test_min[0])))
                                distance.append(float(abs(self.sum_wait_req[job_id]-self.sum_wait_req[j])/float(self.sum_test_max[1]-self.sum_test_min[1])))
                                distance.append(float(abs(self.sum_wait_wallclock[job_id]-self.sum_wait_wallclock[j])/float(self.sum_test_max[2]-self.sum_test_min[2])))
                                distance.append(float(abs(self.sum_run_ert[job_id]-self.sum_run_ert[j])/float(self.sum_test_max[3]-self.sum_test_min[3])))
                                distance.append(float(abs(self.sum_run_req[job_id]-self.sum_run_req[j])/float(self.sum_test_max[4]-self.sum_test_min[4])))
                                distance.append(float(abs(self.sum_run_wallclock[job_id]-self.sum_run_wallclock[j])/float(self.sum_test_max[5]-self.sum_test_min[5])))
                                '''
                                if self.ENABLE_SUMMARY_FEATURES:
                                    this_feature_list.append(self.sum_wait_ert[h_id])
                                    this_feature_list.append(self.sum_wait_req[h_id])
                                    this_feature_list.append(self.sum_wait_elapsed[h_id])

                                    this_feature_list.append(self.sum_run_ert[h_id])
                                    this_feature_list.append(self.sum_run_req[h_id])
                                    this_feature_list.append(self.sum_run_elapsed[h_id])

				    this_feature_list.append(self.sum_higher_priority_jobs[h_id])
                                    this_feature_list.append(self.sum_starving_jobs[h_id])
                                
                                #print h_id, "fl\t",this_feature_list
                                correlation_features.append(this_feature_list)
				if job_id == 53706:
				    print "validation:For job_id=53706 and history_job=",h_id
				    print "validation:this_feature_list=",this_feature_list
				    print "validatin:__wait_time=",self.__wait_time[h_id]
				    print "Validations: history_jobs_length:",len(my_his)
                                correlation_targets.append(self.__wait_time[h_id])
                
		
		if job_id == 53706:
        	    print "Job id",job_id,"History\t",my_his
		correlation_list = []
                spearman_list = []
                #li_Weights = [0, 0, 1, 0.082360000000000003, 0.32868999999999998, 0.01847, 0.055149999999999998, 0.0, 0.1183, 0.38507000000000002, 0.0, 0.0, 0.0, 0.74807999999999997, 0.95338999999999996, 0.0]
                feature_count = 12
                if self.ENABLE_SUMMARY_FEATURES:
                    feature_count += 8
 		for k in range(feature_count):
			# process kth feature
			X = []
			Y = correlation_targets
			for i in range(len(correlation_features)):
				X.append(correlation_features[i][k])
                        if len(X) == 0:
                            correlation_list.append(0.0)
                            continue
			correlation_value = spearmanr(X, Y)

                        if math.isnan(correlation_value[0]): #or abs(correlation_value[0]) < 0.1:
                            correlation_list.append(0.0)
                        else:
                            correlation_list.append(abs(correlation_value[0]))
                if sum(correlation_list) == 0.0:
                    correlation_list = [1.0]*len(correlation_list)
                correlation_list_regression = correlation_list[:]

                #correlation_list[0] = 0.9
                correlation_list[2] = 0.0 # arrival time
                correlation_list[3] = 0.0 # free nodes

                #test_distr_weights = [1.0]*DCount
                
                max_q_n_p_corr = max(correlation_list[4:12])
                ninety_percentage_limit = 0.9*max_q_n_p_corr
                for k in range(4, 12):
                    if correlation_list[k] < ninety_percentage_limit:
                       correlation_list[k] = 0.0 
                
                #correlation_list.extend(distr_correlation_output)

                #print 'features job:', correlation_list[0:4]
                #print 'features user queue:', correlation_list[4:8]
                #print 'features user proc:', correlation_list[8:12]
                #print 'resource state:', correlation_list[12:]
                #print 'Li resource:', correlation_list[12:]

                if self.ENABLE_DISTRIBUTIONS:
                    correlation_list.extend(distr_correlation_output)

                #print 'distr:', correlation_list[12+7:]
                #print 'resource_state:', correlation_list[12:]
                #sys.exit(0)
                
		##
		# Compute component wise distances for each job -- begin
		##
                #num_feat = len(correlation_list)
                #if self.UNWEIGHTED:
                #    correlation_list = [1.0]*num_feat

                total_history_count = 0
                total_same_req_count = 0
                #print 'weights:', correlation_list
		#printValidations
		#if job_id == 63652:
		    #print "Validations: history_jobs_length:",len(self.__history_jobs)
		    #print "Validations: history_jobs=",self.__history_jobs
		for j in self.__history_jobs:	
			if self.__arr_time[job_id] >= (self.__arr_time[j] + self.__wait_time[j]):
				if self.__queue_id[job_id] != self.__queue_id[j]:
					continue
				distance = []
                                total_history_count += 1
				### Job features ###
				'''
				if self.__grp_id[job_id] == self.__grp_id[j]:
					distance.append(0)
				else:
					distance.append(1)					
				if self.__user_id[job_id] == self.__user_id[j]:
					distance.append(0)
				else:
					distance.append(1)

				if self.__queue_id[job_id] == self.__queue_id[j]:
					distance.append(0)
				else:
					distance.append(1)
				if self.__exec_id[job_id] == self.__exec_id[j]:
					distance.append(0)
				else:
					distance.append(1)
                                '''
                                
				if self.__req_size[job_id] == self.__req_size[j]:
					distance.append(0)
                                        total_same_req_count += 1
				else:
				        distance.append(1)
                                
                                  
                                #req_dist = float(abs(self.__req_size[job_id]-self.__req_size[j]))/self.Max_Nodes
                                #req_dist = req_dist*req_dist
				#distance.append(req_dist)
                                ert_dist = float(abs(self.__est_runtime[job_id] - self.__est_runtime[j]))/float(self.__ert_max - self.__ert_min)
                                #ert_dist = ert_dist*ert_dist
				distance.append(ert_dist)
                                ## arr time is disabled via correlation weights, so is free nodes
                                
				#distance.append(float(abs(self.__arr_time[job_id] - self.__arr_time[j]))/float(self.__arr_max - self.__arr_min))
				distance.append(0.0)
				#Prakash: Remove the free nodes feature; IBL GA doesn't consider this - instead 
				#         it has a job attribute feature - queue id (in addition to the policy
				#         attribute queue id)
				#distance.append(float(abs(self.__free_node_map[job_id] - self.__free_node_map[j]))/float(self.Max_Nodes))
                                distance.append(0.0)
				'''
				if self.__queue_id[job_id] == self.__queue_id[j]:
					distance.append(0)
				else:
					distance.append(1)
				'''
				### Li style feature distances ###
				'''

				temp_distance_num = temp_distance_deno = 0
				for k in range(self.__no_queues):
					temp_distance_num += float(abs(self.__proc_occ_no_jobs[job_id][k] - self.__proc_occ_no_jobs[j][k]))
					temp_distance_deno += float(self.__proc_occ_no_jobs[job_id][k] + self.__proc_occ_no_jobs[j][k])
				if temp_distance_deno != 0:
					distance.append(temp_distance_num/temp_distance_deno)
				else:
					distance.append(1)
				temp_distance_num = temp_distance_deno = 0
				for k in range(self.__no_queues):
					temp_distance_num += float(abs(self.__queue_occ_no_jobs[job_id][k] - self.__queue_occ_no_jobs[j][k]))
					temp_distance_deno += float(self.__queue_occ_no_jobs[job_id][k] + self.__queue_occ_no_jobs[j][k])
				if temp_distance_deno != 0:
					distance.append(temp_distance_num/temp_distance_deno)
				else:
					distance.append(1)
				temp_distance_num = temp_distance_deno = 0
				for k in range(self.__no_queues):
					temp_distance_num += float(abs(self.__proc_occ_elapsed_time_total[job_id][k] - self.__proc_occ_elapsed_time_total[j][k]))
					temp_distance_deno += float(self.__proc_occ_elapsed_time_total[job_id][k] + self.__proc_occ_elapsed_time_total[j][k])
				if temp_distance_deno != 0:
					distance.append(temp_distance_num/temp_distance_deno)
				else:
					distance.append(1)
				temp_distance_num = temp_distance_deno = 0
				for k in range(self.__no_queues):
					temp_distance_num += float(abs(self.__proc_occ_remaining_time_total[job_id][k] - self.__proc_occ_remaining_time_total[j][k]))
					temp_distance_deno += float(self.__proc_occ_remaining_time_total[job_id][k] + self.__proc_occ_remaining_time_total[j][k])
				if temp_distance_deno != 0:
					distance.append(temp_distance_num/temp_distance_deno)
				else:
					distance.append(1)
				temp_distance_num = temp_distance_deno = 0
				for k in range(self.__no_queues):
					temp_distance_num += float(abs(self.__queue_occ_elapsed_time_total[job_id][k] - self.__queue_occ_elapsed_time_total[j][k]))
					temp_distance_deno += float(self.__queue_occ_elapsed_time_total[job_id][k] + self.__queue_occ_elapsed_time_total[j][k])
				if temp_distance_deno != 0:
					distance.append(temp_distance_num/temp_distance_deno)
				else:
					distance.append(1)

				temp_distance_num = temp_distance_deno = 0
				for k in range(self.__no_queues):
					temp_distance_num += float(abs(self.__queue_occ_demand_time_total[job_id][k] - self.__queue_occ_demand_time_total[j][k]))
					temp_distance_deno += float(self.__queue_occ_demand_time_total[job_id][k] + self.__queue_occ_demand_time_total[j][k])
				if temp_distance_deno != 0:
					distance.append(temp_distance_num/temp_distance_deno)
				else:
					distance.append(1)
				### Li style feature distances end ###
				'''
                                '''
				### Metric style feature distances ###

				temp_distance_num = temp_distance_deno = 0
				temp_dist = 0.0
				for k in range(self.__no_queues):
					temp_distance_num = float(abs(self.__proc_occ_no_jobs[job_id][k] - self.__proc_occ_no_jobs[j][k]))
					temp_distance_deno = float(self.__proc_occ_no_jobs[job_id][k] + self.__proc_occ_no_jobs[j][k])
					if temp_distance_deno != 0:
						temp_dist += temp_distance_num/temp_distance_deno
					else:
						temp_dist += 0.0
				temp_dist = temp_dist*float(1.0/self.__no_queues)
				distance.append(temp_dist)

				temp_distance_num = temp_distance_deno = 0
				temp_dist = 0.0
				for k in range(self.__no_queues):
					temp_distance_num = float(abs(self.__queue_occ_no_jobs[job_id][k] - self.__queue_occ_no_jobs[j][k]))
					temp_distance_deno = float(self.__queue_occ_no_jobs[job_id][k] + self.__queue_occ_no_jobs[j][k])
					if temp_distance_deno != 0:
						temp_dist += temp_distance_num/temp_distance_deno
					else:
						temp_dist += 0.0
				temp_dist = temp_dist*float(1.0/self.__no_queues)
				distance.append(temp_dist)

				temp_distance_num = temp_distance_deno = 0
				temp_dist = 0.0
				for k in range(self.__no_queues):
					temp_distance_num = float(abs(self.__proc_occ_elapsed_time_total[job_id][k] - self.__proc_occ_elapsed_time_total[j][k]))
					temp_distance_deno = float(self.__proc_occ_elapsed_time_total[job_id][k] + self.__proc_occ_elapsed_time_total[j][k])
					if temp_distance_deno != 0:
						temp_dist += temp_distance_num/temp_distance_deno
					else:
						temp_dist += 0.0
				temp_dist = temp_dist*float(1.0/self.__no_queues)
				distance.append(temp_dist)

				temp_distance_num = temp_distance_deno = 0
				temp_dist = 0.0
				for k in range(self.__no_queues):
					temp_distance_num = float(abs(self.__proc_occ_remaining_time_total[job_id][k] - self.__proc_occ_remaining_time_total[j][k]))
					temp_distance_deno = float(self.__proc_occ_remaining_time_total[job_id][k] + self.__proc_occ_remaining_time_total[j][k])
					if temp_distance_deno != 0:
						temp_dist += temp_distance_num/temp_distance_deno
					else:
						temp_dist += 0.0
				temp_dist = temp_dist*float(1.0/self.__no_queues)
				distance.append(temp_dist)

				temp_distance_num = temp_distance_deno = 0
				temp_dist = 0.0
				for k in range(self.__no_queues):
					temp_distance_num = float(abs(self.__queue_occ_elapsed_time_total[job_id][k] - self.__queue_occ_elapsed_time_total[j][k]))
					temp_distance_deno = float(self.__queue_occ_elapsed_time_total[job_id][k] + self.__queue_occ_elapsed_time_total[j][k])
					if temp_distance_deno != 0:
						temp_dist += temp_distance_num/temp_distance_deno
					else:
						temp_dist += 0.0
				distance.append(temp_dist)					
				temp_distance_num = temp_distance_deno = 0

				temp_dist = 0.0
				for k in range(self.__no_queues):
					temp_distance_num = float(abs(self.__queue_occ_demand_time_total[job_id][k] - self.__queue_occ_demand_time_total[j][k]))
					temp_distance_deno = float(self.__queue_occ_demand_time_total[job_id][k] + self.__queue_occ_demand_time_total[j][k])
					if temp_distance_deno != 0:
						temp_dist += temp_distance_num/temp_distance_deno
					else:
						temp_dist += 0.0
				temp_dist = temp_dist*float(1.0/self.__no_queues)
				distance.append(temp_dist)
				### Metric style feature distances end ###
				

				if self.__grp_id[job_id] == self.__grp_id[j]:
					distance.append(0)
				else:
					distance.append(1)					
				if self.__user_id[job_id] == self.__user_id[j]:
					distance.append(0)
				else:
					distance.append(1)
				'''

				### Submitting user feature distances ###
				# 1. No. of jobs submitted by the user in the current queue 
				# 2. Sum of CPU hours requested by the user in the current queue
				# 3. Sum of ERTs requested by the user in the current queue
				# 4. Sum of number of nodes requested by the user in the current queue
                                
				self.user_queue_waiting_jobs[job_id] = 0.0
				self.user_queue_demand_wallclock_sum[job_id] = 0.0
				self.user_queue_demand_reqsize_sum[job_id] = 0.0
				self.user_queue_demand_ert_sum[job_id] = 0.0
								
				### Final distance computation ###
				feature_id = 0
                                #UQ_waiting_job_distance = 0
                                #UQ_demand_wallclock_distance = 0
                                #UQ_demand_reqsize_distance = 0
                                #UQ_demand_ert_distance = 0
                                #print 'UQFM:', self.user_queue_features_max[feature_id], self.user_queue_features_min[feature_id]

                                feature_name = self.user_queue_waiting_jobs
                                #if self.user_queue_features_max[feature_id]-self.user_queue_features_min[feature_id] != 0:
				#if job_id == 5068 and j == 4096:
				if job_id == 53706:
				    print "Validation: feature vals: job_id:",job_id," j=",j," feature name=",feature_name
				    print "feature_name[job_id]=",feature_name[job_id],"feature_name[j]=",feature_name[j]," self.user_queue_features_max[feature_id]=",self.user_queue_features_max[feature_id]," self.user_queue_features_min[feature_id]=",self.user_queue_features_min[feature_id]
                                UQ_waiting_job_distance = float(abs(feature_name[job_id]-feature_name[j]))/float(self.user_queue_features_max[feature_id]-self.user_queue_features_min[feature_id])                                 
                                feature_id = 1
                                feature_name = self.user_queue_demand_wallclock_sum
				#if job_id == 5068 and j == 4096:
				if job_id == 53706:
				    print "Validation: feature vals: job_id:",job_id," j=",j," feature name=",feature_name
				    print "feature_name[job_id]=",feature_name[job_id],"feature_name[j]=",feature_name[j]," self.user_queue_features_max[feature_id]=",self.user_queue_features_max[feature_id]," self.user_queue_features_min[feature_id]=",self.user_queue_features_min[feature_id]
                                #if self.user_queue_features_max[feature_id]-self.user_queue_features_min[feature_id] != 0:
                                UQ_demand_wallclock_distance = float(abs(feature_name[job_id]-feature_name[j]))/float(self.user_queue_features_max[feature_id]-self.user_queue_features_min[feature_id])
                                feature_id = 2
                                feature_name = self.user_queue_demand_reqsize_sum
                                #if self.user_queue_features_max[feature_id]-self.user_queue_features_min[feature_id] != 0:
                                UQ_demand_reqsize_distance = float(abs(feature_name[job_id]-feature_name[j]))/float(self.user_queue_features_max[feature_id]-self.user_queue_features_min[feature_id])
                                feature_id = 3
                                feature_name = self.user_queue_demand_ert_sum
                                #if self.user_queue_features_max[feature_id]-self.user_queue_features_min[feature_id] != 0:
                                UQ_demand_ert_distance = float(abs(feature_name[job_id]-feature_name[j]))/float(self.user_queue_features_max[feature_id]-self.user_queue_features_min[feature_id])
                                #feature_id = 4
                                #feature_name = self.user_queue_elapsed_wallclock_sum
                                #UQ_elapsed_wallclock_distance = float(abs(feature_name[job_id]-feature_name[j]))/float(self.user_queue_features_max[feature_id]-self.user_queue_features_min[feature_id])


				distance.append(UQ_waiting_job_distance)
				distance.append(UQ_demand_wallclock_distance)
				distance.append(UQ_demand_reqsize_distance)
				distance.append(UQ_demand_ert_distance)
				#distance.append(UQ_elapsed_wallclock_distance)

				feature_id = 0
				feature_name = self.user_proc_running_jobs
				UR_running_job_distance = float(abs(feature_name[job_id]-feature_name[j]))/float(self.user_proc_features_max[feature_id]-self.user_proc_features_min[feature_id])
				feature_id = 1 
				feature_name = self.user_proc_demand_wallclock_sum
				UR_demand_wallclock_distance = float(abs(feature_name[job_id]-feature_name[j]))/float(self.user_proc_features_max[feature_id]-self.user_proc_features_min[feature_id])
				feature_id = 2
				feature_name = self.user_proc_demand_reqsize_sum
				UR_demand_reqsize_distance = float(abs(feature_name[job_id]-feature_name[j]))/float(self.user_proc_features_max[feature_id]-self.user_proc_features_min[feature_id])
				feature_id = 3
				feature_name = self.user_proc_demand_ert_sum
				UR_demand_ert_distance = float(abs(feature_name[job_id]-feature_name[j]))/float(self.user_proc_features_max[feature_id]-self.user_proc_features_min[feature_id])
                                #feature_id = 4
                                #feature_name = self.user_proc_elapsed_wallclock_sum
                                #UR_elapsed_wallclock_distance = float(abs(feature_name[job_id]-feature_name[j]))/float(self.user_proc_features_max[feature_id]-self.user_proc_features_min[feature_id])

                                distance.append(UR_running_job_distance)
                                distance.append(UR_demand_wallclock_distance)
                                distance.append(UR_demand_reqsize_distance)
                                distance.append(UR_demand_ert_distance)
                                #distance.append(UR_elapsed_wallclock_distance)
                                


                                if self.ENABLE_SUMMARY_FEATURES:
                                    #print 'STM:',  self.sum_test_max[0], self.sum_test_min[0]
                                    #if self.sum_test_max[0] != self.sum_test_min[0]:
                                        # Every job upto now has had same sum of ERT values - dont consider this feature
                                    distance.append(float(abs(self.sum_wait_ert[job_id]-self.sum_wait_ert[j])/float(self.sum_test_max[0]-self.sum_test_min[0])))
                                    #else:
                                    #distance.append(0.0)

                                    #if self.sum_test_max[1] != self.sum_test_min[1]:
                                    distance.append(float(abs(self.sum_wait_req[job_id]-self.sum_wait_req[j])/float(self.sum_test_max[1]-self.sum_test_min[1])))
                                    #else:
                                    #distance.append(0.0)
                                    #if self.sum_test_max[2] != self.sum_test_min[2]:
                                    distance.append(float(abs(self.sum_wait_elapsed[job_id]-self.sum_wait_elapsed[j])/float(self.sum_test_max[2]-self.sum_test_min[2])))
                                    #else:
                                    #distance.append(0.0)
                                    #if self.sum_test_max[3] != self.sum_test_min[3]:                                        
                                    distance.append(float(abs(self.sum_run_ert[job_id]-self.sum_run_ert[j])/float(self.sum_test_max[3]-self.sum_test_min[3])))
                                    #else:
                                    #distance.append(0.0)
                                    #if self.sum_test_max[4] != self.sum_test_min[4]:
                                    distance.append(float(abs(self.sum_run_req[job_id]-self.sum_run_req[j])/float(self.sum_test_max[4]-self.sum_test_min[4])))
                                    #else:
                                    #distance.append(0.0)
                                    #if self.sum_test_max[5] != self.sum_test_min[5]:
                                    distance.append(float(abs(self.sum_run_elapsed[job_id]-self.sum_run_elapsed[j])/float(self.sum_test_max[5]-self.sum_test_min[5])))

				    distance.append(float(abs(self.sum_higher_priority_jobs[job_id]-self.sum_higher_priority_jobs[j])/float(self.sum_test_max[6]-self.sum_test_min[6])))

				    distance.append(float(abs(self.sum_starving_jobs[job_id]-self.sum_starving_jobs[j])/float(self.sum_test_max[7]-self.sum_test_min[7])))
                                    #else:
                                    #distance.append(0.0)
				### Distribution Distances ###
				if self.ENABLE_DISTRIBUTIONS: 
                                    distance.extend(distribution_distance[j])
                                
				#print '#features in distance computation=', len(distance)
				# We've got all the necessary distances for this job
				# Append it to a list --> this list will be used by each predictor
				individual_distance_map[j] = distance
                                #validation
                                #if job_id == 5068:
                                    #print "Validation: For j=",j," individual_distance_map=",distance
				individual_wait_times[j] = self.__wait_time[j]

		# Compute component wise distances for each job -- end
	        					
		predicted_wait = 0

                if len(individual_distance_map) < len(correlation_list):
                    self.__ibl_predicted_wait_time[job_id] = 0                    
                    return 0
                #print job_id, '20-WA', wa_predicted_wait
                
                pred_list = []
                dmap = []
                go_ahead = False
                if not self.ENABLE_DISTRIBUTIONS:
                    # do sd scan with features first
		    #printValidation
	            #print "Validation: model: job_id=",job_id," pred 3_Web"
                    dbscan_pred_output, pred_list, dmap, invalid_output = self.dbscan_pred.get_prediction_plain(individual_distance_map, individual_wait_times, job_id, correlation_list,self.__wait_time[job_id])
                    go_ahead = invalid_output # invalid_output is true if we require regression
                    print 'Do_regression:', invalid_output

                    if not invalid_output:
                        predicted_wait = int(dbscan_pred_output)
                if go_ahead or self.ENABLE_DISTRIBUTIONS:
                    pred_list, dmap = self.dbscan_pred.preprocess_for_regression(individual_distance_map, individual_wait_times, job_id, correlation_list,self.__wait_time[job_id])
                    print 'Did preprocess for regression'
                    #print "PRED",pred_list
                #prediction_str = str(int(predicted_wait))
                if go_ahead or self.ENABLE_DISTRIBUTIONS:
                    #outlier_list = self.dbscan_pred.get_global_outliers()
                    outlier_list = []
                    key_list = []
                    print 'Pred_list:', len(pred_list), 'reg distance =', test_regression_distance

                    current_regression_boundary = test_regression_distance
                    print 'TRD:', test_regression_distance, 'correlations=', len(correlation_list)

                    key_list = []
                    while current_regression_boundary <= 1.0:
                        #print 'here in the while loop', current_regression_boundary
                        current_key_list = []
                        for item in pred_list:
                            if not item[2] in outlier_list and dmap[item[2]] < current_regression_boundary:
                                current_key_list.append(item[2])
                        #print 'current key list size=', len(current_key_list)
                        if len(current_key_list) > 10*len(correlation_list):
                            key_list = current_key_list
                            break
                        current_regression_boundary += 0.1
                    key_list.append(job_id) # this is for the normalization part

                    min_points = 10
                    regression_prediction = 0.0
                    regression_valid = False
                    #print 'keys:', len(key_list), 'minpts:', min_points
                    if len(key_list) >= min_points:
                        # get feature vectors of jobs in list
                        regression_valid = True
                        #print len(key_list), key_list
                        XREG = []
                        YREG = []
                        #WEIGHT_ARRAY = []
                        for key in key_list:
                            #print self.job_feature_vector[key]
                            relevant_features = []
                            # do a regression on the relevant features only <- this is where we use the correlation weights 
                            for cindex in range(len(correlation_list_regression)):
                                if correlation_list_regression[cindex] > 0:
                                    relevant_features.append(self.job_feature_vector[key][cindex])
                            #relevant_features.append(1.0) # constant term
                            XREG.append(relevant_features)
                            #print relevant_features
                            if key != job_id:
                                YREG.append(self.__wait_time[key])
                                #WEIGHT_ARRAY.append(math.exp((1.0-dmap[key])**2.0))

                        XREG_NP = numpy.array(XREG)
                        XREG_NP_scaled = preprocessing.scale(XREG_NP)
                        #print len(XREG_NP_scaled[0])
                        #print "YREG: ",YREG

                        # fit model
                        #model = sm.WLS(YREG, XREG_NP_scaled[:-1], weights=WEIGHT_ARRAY)
                        #results = model.fit()
                        #regression_output = results.predict(XREG_NP_scaled[-1])
                        #wls_r_value = model.score(XREG_NP_scaled[:-1])
                        #print results.summary()                        
                        #print 'Reg:', regression_output

                        '''
                        current_reg_model.fit(XREG_NP_scaled[:-1], YREG)
                        r_value = current_reg_model.score(XREG_NP_scaled[:-1], YREG)
                        regression_output = current_reg_model.predict(XREG_NP_scaled[-1])
                        print '*Reg output=', regression_output, 'score=', r_value

                        try:
                            lasso_reg_model.fit(XREG_NP_scaled[:-1], YREG)
                            r_value = lasso_reg_model.score(XREG_NP_scaled[:-1], YREG)
                            lasso_regression_output = lasso_reg_model.predict(XREG_NP_scaled[-1])
                            #print '\n*Lasso output=', regression_output, 'score=', r_value
                        except Exception:
                            lasso_regression_output = -1
                            pass
                        '''
                        try:
                            ridge_reg_model.fit(XREG_NP_scaled[:-1], YREG)
                            r_value = ridge_reg_model.score(XREG_NP_scaled[:-1], YREG)
                            ridge_regression_output = ridge_reg_model.predict(XREG_NP_scaled[-1])
                            #cnum = COND_NUMBER(XREG_NP_scaled)
                            #print 'Condition number:', cnum
                            #print job_id, 'Ridge output=', ridge_regression_output, 'score=', r_value
                        except Exception:
                            ridge_regression_output = -1
                            pass
                        regression_output = ridge_regression_output
                        '''
                        try:
                            dt_reg_model.fit(XREG_NP_scaled[:-1], YREG)
                            r_value = dt_reg_model.score(XREG_NP_scaled[:-1], YREG)
                            regression_output = dt_reg_model.predict(XREG_NP_scaled[-1])
                            dt_regression_output = regression_output[0]
                            #print '*T output=', regression_output, 'score=', r_value
                        except Exception:
                            dt_regression_output = -1
                            pass
                        
                        regression_output = ridge_regression_output

                        #print job_id, 'lasso:', lasso_regression_output, 'ridge:', ridge_regression_output, 'dt:', dt_regression_output
                        '''
                        if regression_output > 0:
			    if job_id == 53706:
 			    	print "validation:\t", job_id, "\t","Regression output for 53706:",regression_output
                            predicted_wait = regression_output                            
                            #prediction_str += ' R= '+ str(int(regression_output))
                        else:
                            # Dont use the regression ouput
                            if len(pred_list):
                                predicted_wait = self.WA_Predictor.get_prediction_plain(individual_distance_map, individual_wait_times, job_id, correlation_list)
                                #print job_id, 'WA output=', predicted_wait
				if job_id == 53706:
 			    		print "validation:\t", job_id, "\t","WA predictor output for 53706:", predicted_wait
                            else:
                                predicted_wait = 10.0


                self.__ibl_predicted_wait_time[job_id] = predicted_wait                    
                if predicted_wait != -1:
                    #print job_id, 'actual=', self.__wait_time[job_id], 'pred=', predicted_wait, 'run=', self.__real_run_time[job_id]
                    #print "Prediction 2"
                    #print "Hi from final part of prediction"
		    #if job_id == 41235:
			#print "Predicted Wait Time for 41235\t",predicted_wait

                    self.PredictedWaitingTime[job_id] = predicted_wait
		    #result_file.write(str(job_id))
		    #result_file.write("\t")
		    #result_file.write(str(predicted_wait))
		    #result_file.write("\n")
		    

                #print 'HCount:', total_history_count, 'SReq:', total_same_req_count, '%', 100.0*total_same_req_count/total_history_count
                
                if 0 and job_id in check_list:
                    '''
                    print ""
                    print "----------------"
                    print pred_list[:20]
                    print "----------------"
                    print ""
                    for item in pred_list[:20]:
                        item_id = item[2]
                        print item_id, item[1], self.job_feature_vector[item_id]

                    print job_id, 'actual=', self.__wait_time[job_id], 'pred=', predicted_wait, 'run=', self.__real_run_time[job_id]
                    print 'R:', self.__req_size[job_id], 'ERT:', self.__est_runtime[job_id], 'U:', self.__user_id[job_id], 'E:', self.__exec_id[job_id], 'G:', self.__grp_id[job_id], 'Run:', self.__real_run_time[job_id], 'Q:', self.__queue_id[job_id]
                    #print 'WA_Pred=', wa_predicted_wait
                    '''
                    pX = []
                    pY = []
                    
                    x_percent = 1.0
                    top_x_percent = int(x_percent*len(pred_list))
                    count_x_percent = 0



                    while count_x_percent < top_x_percent:
                        item = pred_list[count_x_percent]
                        pX.append(item[0])
                        pY.append(item[1] - self.__wait_time[job_id])
                        count_x_percent += 1

                    
                    
                    #print len(pX), len(pY)
                    #print pX[:100], pY[:100]
                    
                    plt.clf()
                    plt.scatter(pX,pY)
                    #title = 'Job id: ' + str(job_id) + ' Wait time= ' + str(self.__wait_time[job_id]) + ' D= ' + prediction_str # + ' Pred=' + str(predicted_wait)
                    title = str(job_id)
                    plt.title(title)
                    plt.xlabel('distance')
                    plt.ylabel('var. wait time')
                    #plt.show()
                    name = 'TyroneFeat' + str(job_id) + '.png'
                    #plt.show()
                    plt.savefig(name)
                
		return predicted_wait
		
	# Function to predict the Queue Wait Times
	def __job_qwt_predictor(self, job_id):		
		slab_size = [ 900, 1800, 3600, 10800, 21600, 43200, 86400 ]
                #print job_id,"from job qwt\t",self.__wait_time[job_id]
		#self.__wait_time[job_id] = self.__start_time[job_id] - self.__arr_time[job_id]
		upper_bound = self.__get_qwt_strategy_based(job_id, self.__wait_time[job_id]) #_wait_time refer line 2585 and 2598
		runtime = self.__round_time(self.__real_run_time[job_id],1)   #__real_run_time line no 2537
		waittime = self.__round_time(self.__wait_time[job_id],1)
		ibl_waittime = self.__round_time(self.__ibl_predicted_wait_time[job_id],1)

		#print 'job', job_id,'qwt=',waittime
		if waittime <= ibl_waittime:
			in_range = 1
		else:
			in_range = 0
	
		blk_diff_ub = -1
		if abs(ibl_waittime - waittime) <= 3600 or abs(ibl_waittime - waittime) <= slab_size[self.__getslab(ibl_waittime)]:
			blk_diff_ub = 0	
			self.__good_metric_ibl += 1			
			self.__metric_list_ibl.append(blk_diff_ub)
			self.__metric_list_ibl_map[job_id] = blk_diff_ub
			in_range = 1
			
		else:
			blk_diff_ub = abs(ibl_waittime - waittime)/float(runtime + waittime)
			self.__metric_list_ibl.append(blk_diff_ub)
			self.__metric_list_ibl_map[job_id] = blk_diff_ub
			if blk_diff_ub <= 0.1:
				self.__good_metric_ibl += 1		
		#if job_id % 1000:
		#	x=numpy.array(self.__metric_list_ibl)
		#	print 'Current avg=', numpy.mean(x)
		return in_range
	

	

	# Convert the SWF Log into suitable format
	def convert_logs(self, input_file_name):		
		est_runtime = {} ; req_size = {} ; event_list = []
		job_id = 1

		#  Read the file & create the list
		file_input_log = open(input_file_name, 'r')
		file_converted_log = open("job.txt", 'w')
		file_converted_stats = open("timeinfo.txt", 'w')
		lines = file_input_log.readlines()

		for line in lines:
			job = line.split()
			
			if int(job[2]) < 0:
				continue
			if int(job[3]) < 0:
				continue
			if int(job[4]) <= 0:
				continue
			if int(job[8]) < 0:
				job[8] = "-1"
				
			req_size[job_id] = int(job[4])
			est_runtime[job_id] = int(job[8])
			event_list.append((int(job[1]), job_id, int(job[14]), 1))
			event_list.append((int(job[1]) + int(job[2]), job_id, int(job[14]), 2))
			event_list.append((int(job[1]) + int(job[2]) + int(job[3]), job_id, int(job[14]), 3))
			
			file_converted_stats.write(str(job_id) + " " + job[2] + " " + job[3] + " " + job[11] + " " + job[12] + " " + job[13] + "\n")
			
			job_id += 1
			
		file_input_log.close()
		
		event_list.sort()

		for i in range(len(event_list)):
			file_converted_log.write(str(event_list[i][1]))
			file_converted_log.write(" ")
			file_converted_log.write(str(event_list[i][0]))
			file_converted_log.write(" ")
			file_converted_log.write(str(req_size[event_list[i][1]]))
			file_converted_log.write(" ")
			file_converted_log.write(str(est_runtime[event_list[i][1]]))
			file_converted_log.write(" ")
			file_converted_log.write(str(event_list[i][2]))
			file_converted_log.write(" ")
			file_converted_log.write(str(event_list[i][3]))
			file_converted_log.write("\n")
		file_converted_log.close()
		file_converted_stats.close()
        '''
        Arrival event notification comes from the meta scheduler
        '''
        def notify_arrival_event(self, inJob, waiting_jobs, running_jobs):
            # use the waiting and running(not actually needed) provided from args
            #self.__running_job_id = [job.id for job in running_jobs]
            #print "running jobs coming from notify_arrival_event", self.__running_job_id
            self.__waiting_job_id = [job.id for job in waiting_jobs]
	    #if inJob.id == 53706:
	    #	print "Validation:\tWaiting jobs for 53706\t",self.__waiting_job_id
            #print "Job id",job.id,"\t","Waiting job ids from notify_arrival_event",self.__waiting_job_id
            self.__waiting_job_ert = [job.user_estimated_run_time for job in waiting_jobs]
            #if inJob.id == 53706:
	    #	print "Validation:\tWaiting jobs ert for 53706\t",self.__waiting_job_ert
            #print "Job id",job.id,"\t","User estimated run time from notify_arrival_event",self.__waiting_job_ert
            self.__waiting_job_reqsize = [job.num_required_processors for job in waiting_jobs] 
            #if inJob.id == 53706:
	    #	print "Validation:\tWaiting jobs reqsize for 53706\t",self.__waiting_job_reqsize
            #print "Job id",job.id,"\t","User estimated run time from notify_arrival_event",self.__waiting_job_reqsize
            self.__waiting_job_cpu = [job.num_required_processors*job.user_estimated_run_time for job in waiting_jobs]
            #if inJob.id == 53706:
	    #	print "validation\t Waiting jobs cpu for 53706\t",self.__waiting_job_cpu
            #print "Job id",job.id,"\t","User estimated run time from notify_arrival_event",self.__waiting_job_cpu
            
            self.__arr_time[inJob.id] = inJob.submit_time
            #if inJob.id == 53706:
	    #	print "validation\t Arrival time for 53706\t",self.__arr_time[inJob.id]
            #print "Job id",job.id,"\t","Arrival time of job id", self.__arr_time[inJob.id]
            #self.__req_size[inJob.id] = inJob.num_required_processors
            self.__req_size[inJob.id] = inJob.num_requested_processors
            #if inJob.id == 53706:
	    #	print "validation\t Req size for 53706\t",self.__req_size[inJob.id]
            #print "Job id",job.id,"\t","Arrival time of job id",self.__req_size[inJob.id]
            self.__est_runtime[inJob.id] = inJob.user_estimated_run_time
            #if inJob.id == 53706:
	    #	print "validation\t est for 53706\t",self.__est_runtime[inJob.id]
            #print "Job id",job.id,"\t","est runtime of job id", self.__est_runtime[inJob.id]
            #if inJob.id == 5001:
		#print "est for",inJob.id,"\t",self.__est_runtime[inJob.id]
            self.__queue_id[inJob.id] = inJob.queue_id
	    #cray
            self.__queue_group[inJob.id] = inJob.queue_group
	    
            #if inJob.id == 53706:
	    #	print "validation\t queue id for 53706\t", self.__queue_id[inJob.id]
            #if inJob.id == 53706:
	    #	print "validation\t queue group for 53706\t", self.__queue_group[inJob.id]
            #print "Job id",job.id,"\t","est runtime of job id", self.__queue_id
            self.__real_run_time[inJob.id] = inJob.actual_run_time
            #if inJob.id == 53706:
	    #	print "validation\treal run time for 53706\t",  self.__real_run_time[inJob.id]
            #print "Job id",job.id,"\t","est runtime of job id", self.__real_run_time[inJob.id]
            self.__user_id[inJob.id] = inJob.user_id
            #if inJob.id == 53706:
	    #	print "validation\tuser id for 53706\t", self.__user_id[inJob.id]
            #print "Job id",job.id,"\t","est runtime of job id", self.__user_id
            
            self.__grp_id[inJob.id] = -1                   # TODO
            self.__exec_id[inJob.id] = -1                  # TODO
                     
            my_job = inJob.id
            self.__ert_min = min(self.__ert_min, self.__est_runtime[my_job])
            self.__ert_max = max(self.__ert_max, self.__est_runtime[my_job])
            self.__arr_min = min(self.__arr_min, self.__arr_time[my_job])
            self.__arr_max = max(self.__arr_max, self.__arr_time[my_job])
            self.__req_min = min(self.__req_min, self.__req_size[my_job])
            self.__req_max = max(self.__req_max, self.__req_size[my_job])
            #self.__history_jobs.insert(0,inJob.id) 
            #if int(inJob.id) <= test_target_start_point_in_trace:
            #self.__history_jobs.insert(0,inJob.id)
            #if len(self.__history_jobs) > 5000:
               #self.__history_jobs.insert(0,inJob.id)
               #self.__history_jobs.pop()
            #if inJob.id == 5009: 
               #print inJob.id,"\t",self.__history_jobs  
            #print "job:",inJob.id,"\t",self.__history_jobs
	    #print inJob.id, "\n", "__waiting_job_id", self.__waiting_job_id, "\nrunning_job_id", self.__running_job_id	#sid
            self.__set_job_properties(inJob.id, inJob.num_required_processors, inJob.user_estimated_run_time)

        def delete_tentative_arrival_event(self, inJob):

            del self.__arr_time[inJob.id]
            del self.__req_size[inJob.id]
            del self.__est_runtime[inJob.id]
            del self.__queue_id[inJob.id]

            del self.__real_run_time[inJob.id]
            del self.__user_id[inJob.id]
            del self.__grp_id[inJob.id]
            del self.__exec_id[inJob.id]

        '''
        Expected use case is one call to notify_arrival_event and multiple calls to get_prediction
        '''
        def get_prediction(self, inJob):
            # The following three attributes may be modified by the meta scheduler
            #self.__req_size[inJob.id] = inJob.num_required_processors   #_req_size, est_runtime and real_run_time are dictionary
            self.__req_size[inJob.id] = inJob.num_requested_processors
	    if inJob.id == 53706:
            	print "validation\t",inJob.id,"\t",self.__req_size[inJob.id]
            self.__est_runtime[inJob.id] = inJob.user_estimated_run_time
	    if inJob.id == 53706:
		print "validation\tEstimated run time\t",self.__est_runtime[inJob.id]
            self.__real_run_time[inJob.id] = inJob.actual_run_time
	    if inJob.id == 53706:
		print "validation\tReal run time\t",self.__real_run_time[inJob.id]
            my_job = inJob.id
            self.__ert_min = min(self.__ert_min, self.__est_runtime[my_job])
            self.__ert_max = max(self.__ert_max, self.__est_runtime[my_job])
            self.__arr_min = min(self.__arr_min, self.__arr_time[my_job])
            self.__arr_max = max(self.__arr_max, self.__arr_time[my_job])
            self.__req_min = min(self.__req_min, self.__req_size[my_job])
            self.__req_max = max(self.__req_max, self.__req_size[my_job])

            #self.__target_jobs.insert(0,inJob.id)

            # need to update self.job_feature_vector entries
            #no effect on prediction when job feature vector is commented out 
            self.job_feature_vector[my_job][0] = self.__req_size[inJob.id]              
            self.job_feature_vector[my_job][1] = self.__est_runtime[inJob.id]
            self.__wait_time[inJob.id] = round(inJob.start_to_run_at_time - inJob.submit_time, 1)
	    if inJob.id == 53706:
            	print inJob.id,"validation\twait time From get_prediction\t",self.__wait_time[inJob.id]
            #print inJob.id,"\t",self.__start_time[inJob.id]
	    #if inJob.id == 53706:
            #	print inJob.id,"validation:= start time From get_prediction\t",self.__start_time[inJob.id] 
            #self.__wait_time[inJob.id] = self.__start_time[inJob.id] - self.__arr_time[inJob.id]
            #if int(inJob.id) > test_target_start_point_in_trace:
            self.__set_job_properties(inJob.id, inJob.num_required_processors, inJob.user_estimated_run_time)
            self.__job_qwt_predictor(inJob.id)
            #if int(inJob.id) <= test_target_start_point_in_trace:
               #self.__history_jobs.insert(0,inJob.id)
               #print "job:",inJob.id,"\t",self.__history_jobs
               #print inJob.id, "\n", "__waiting_job_id", self.__waiting_job_id, "\nrunning_job_id", self.__running_job_id      #sid
	    if inJob.id in self.PredictedWaitingTime:
		return self.PredictedWaitingTime[inJob.id]
	    else:
		errorFile = open('/home/siddharthsahu/Desktop/Cray_prediction/Cray_11_4/Cray/src/KeyErrorJobs', 'a')
		errorFile.write("Key Error job id: "+str(inJob.id)+"\t\t"+str(inJob.queue_id)+"\n")
		errorFile.close()
        
        def notify_job_start_event(self, inJob, waiting_jobs, running_jobs):
            #print "hi"
            self.__running_job_id = [job.id for job in running_jobs]
	    if inJob.id == 53706:
            	print job.id,"\t","rr from notify",self.__running_job_id
            self.__waiting_job_id =  [job.id for job in waiting_jobs]
	    if inJob.id == 53706:
            	print "job id",inJob.id,"\t","Waiting job ids from notify_job_start_event",self.__waiting_job_id
            self.__waiting_job_ert = [job.user_estimated_run_time for job in waiting_jobs]
	    if inJob.id == 53706:
	    	print "job id",inJob.id,"\t","Waiting job ert from notify_job_start_event",self.__waiting_job_ert 
            self.__waiting_job_reqsize = [job.num_required_processors for job in waiting_jobs]
	    if inJob.id == 53706: 
            	print "job id",inJob.id,"\t","Waiting job req_size from notify_job_start_event",self.__waiting_job_reqsize
            self.__waiting_job_cpu = [job.num_required_processors*job.user_estimated_run_time for job in waiting_jobs]
	    if inJob.id == 53706:
            	print "job id",inJob.id,"\t","Waiting job cpu from notify_job_start_event", self.__waiting_job_cpu
            self.__start_time[inJob.id] = inJob.start_to_run_at_time
	    if inJob.id == 53706: 
            	print inJob.id,"start time\t", self.__start_time[inJob.id]
            
            self.__wait_time[inJob.id] = round(self.__start_time[inJob.id] - self.__arr_time[inJob.id],1)
	    if inJob.id == 53706:
            	print inJob.id,"wait time\t", self.__wait_time[inJob.id]
            	print inJob.id,"Arrival time\t", self.__arr_time[inJob.id]
            	print inJob.id,"wait time from notify\t",self.__wait_time[inJob.id]
            self.__running_job_reqsize = [job.num_required_processors for job in running_jobs]
            self.__running_job_ert = [job.user_estimated_run_time for job in running_jobs]
            self.__running_job_cpu = [job.num_required_processors*job.user_estimated_run_time for job in running_jobs]
	    
            self.__free_nodes = self.Max_Nodes - sum(self.__running_job_reqsize)
         
            self.__wait_time[inJob.id] = self.__start_time[inJob.id] - self.__arr_time[inJob.id]
            self.__elapsed_wait_max = max(self.__elapsed_wait_max, self.__wait_time[inJob.id])
            self.__elapsed_wait_min = min(self.__elapsed_wait_min, self.__wait_time[inJob.id])
            #print self.__running_job_id
            #print self.__history_jobs
            #if int(inJob.id) <= (test_target_start_point_in_trace):
            #print "adding"
            self.__history_jobs.insert(0,inJob.id)
            #self.__set_job_properties(inJob.id, inJob.num_required_processors, inJob.user_estimated_run_time)
           
            if len(self.__history_jobs) >5000:
               #self.__history_jobs.insert(0,inJob.id)
	       temp_job = min(self.__history_jobs) 
               #self.__history_jobs.pop()
	       self.__history_jobs.remove(temp_job)
          
            #if inJob.id == 5009:
              # print "job:",job.id,"History jobs\t",self.__history_jobs
        def notify_job_termination_event(self, inJob, waiting_jobs, running_jobs):
            self.__running_job_id = [job.id for job in running_jobs]
            self.__running_job_reqsize = [job.num_required_processors for job in running_jobs]
            self.__running_job_ert = [job.user_estimated_run_time for job in running_jobs]
            self.__running_job_cpu = [job.num_required_processors*job.user_estimated_run_time for job in running_jobs]
            self.__free_nodes = self.Max_Nodes - sum(self.__running_job_reqsize)
            self.__elapsed_run_max = max(self.__elapsed_run_max, self.__real_run_time[inJob.id])
            self.__elapsed_run_min = min(self.__elapsed_run_min, self.__real_run_time[inJob.id])

            #self.__running_job_id.remove(int(inJob.id))
            #self.__running_job_reqsize.remove(int(inJob.num_required_processors))
            #self.__running_job_ert.remove(int(inJob.user_estimated_run_time))
            #self.__running_job_cpu.remove(int(inJob.num_required_processors*inJob.user_estimated_run_time))
            #self.__free_nodes = self.Max_Nodes - sum(self.__running_job_reqsize)
            #self.__elapsed_run_max = max(self.__elapsed_run_max, self.__real_run_time[inJob.id])
            #self.__elapsed_run_min = min(self.__elapsed_run_min, self.__real_run_time[inJob.id])
            
            #self.__running_job_id.remove(int(job[0]))
            #self.__running_job_reqsize.remove(int(job[2]))
            #self.__running_job_ert.remove(int(job[3]))
            #self.__running_job_cpu.remove(int(job[2]) * int(job[3]))
            #self.__free_nodes += int(job[2])
            #self.__elapsed_run_max = max(self.__elapsed_run_max, self.__real_run_time[int(job[0])])
            #self.__elapsed_run_min = min(self.__elapsed_run_min, self.__real_run_time[int(job[0])])
	# Parse the Job File
	def log_parser(self):
	
		#  Read the file & create the list
		file_input_log = open("job.txt", 'r')
		lines = file_input_log.readlines()
		
		is_target_job = {}
		joblist = [] ; 
		total_in_range = 0 
		
		for line in lines:
			job = line.split()
			joblist.append(job)
		file_input_log.close()
		
		# Read the file & set the run time & wait time
		file_input_stats = open("timeinfo.txt", 'r')
		lines = file_input_stats.readlines()
		for line in lines:
			job = line.split()
			self.__wait_time[int(job[0])] = int(job[1])
			self.__real_run_time[int(job[0])] = int(job[2])
			self.__user_id[int(job[0])] = int(job[3])
			self.__grp_id[int(job[0])] = int(job[4])
			self.__exec_id[int(job[0])] = int(job[5])
		file_input_stats.close()
			
		# Create the current system state lists
		for job in joblist:
			if job[5] == "1" and int(job[2]) != -1:	
				self.__waiting_job_id.append(int(job[0]))
                                #print "job id",job[0],"\t","Waiting job ids",self.__waiting_job_id
				self.__waiting_job_ert.append(int(job[3]))
				self.__waiting_job_reqsize.append(int(job[2]))
				self.__waiting_job_cpu.append(int(job[2]) * int(job[3]))
				
				self.__arr_time[int(job[0])] = int(job[1])
				self.__req_size[int(job[0])] = int(job[2])
				self.__est_runtime[int(job[0])] = int(job[3])
				self.__queue_id[int(job[0])] = int(job[4])

				#self.__ert_min = float('inf')
				#self.__ert_max = -float('inf')
				#self.__arr_min = float('inf')
				#self.__arr_max = -float('inf')
				#for my_job in self.__history_jobs:
				my_job = int(job[0])
				self.__ert_min = min(self.__ert_min, self.__est_runtime[my_job])
				self.__ert_max = max(self.__ert_max, self.__est_runtime[my_job])
				self.__arr_min = min(self.__arr_min, self.__arr_time[my_job])
				self.__arr_max = max(self.__arr_max, self.__arr_time[my_job])
                                self.__req_min = min(self.__req_min, self.__req_size[my_job])
                                self.__req_max = max(self.__req_max, self.__req_size[my_job])

				if int(job[0])>=self.__History_Begin and int(job[0])<=self.__History_End:
					self.__history_jobs.insert(0,int(job[0]))
					self.__set_job_properties(int(job[0]), int(job[2]), int(job[3]))
						
				else:
					if len(self.__target_jobs) < self.__Target_Size:
						if int(job[0]) >= self.__TargetJobs_Begin:
							is_target_job[int(job[0])] = 1
							
							self.__target_jobs.insert(0,int(job[0]))
							self.__set_job_properties(int(job[0]), int(job[2]), int(job[3]))
							in_range = self.__job_qwt_predictor(int(job[0]))
							total_in_range += in_range
						else:
							is_target_job[int(job[0])] = 0
					else:
						is_target_job[int(job[0])] = 0
														
			elif job[5] == "2" and int(job[2]) != -1:
				self.__running_job_id.append(int(job[0]))
				self.__waiting_job_id.remove(int(job[0]))
				self.__waiting_job_ert.remove(int(job[3]))
				self.__waiting_job_reqsize.remove(int(job[2]))
				self.__waiting_job_cpu.remove(int(job[2]) * int(job[3]))
				self.__start_time[int(job[0])] = int(job[1])
				self.__running_job_reqsize.append(int(job[2]))	
				self.__running_job_ert.append(int(job[3]))
				self.__running_job_cpu.append(int(job[2]) * int(job[3]))
				self.__free_nodes -= int(job[2])

				self.__elapsed_wait_max = max(self.__elapsed_wait_max, self.__wait_time[int(job[0])])
                                self.__elapsed_wait_min = min(self.__elapsed_wait_min, self.__wait_time[int(job[0])])

				if int(job[0]) >= self.__TargetJobs_Begin and is_target_job[int(job[0])] == 1:
					self.__history_jobs.insert(0,int(job[0]))
					self.__history_jobs.pop()
					if 0:
						self.Jobs_Since_Last_Score_Computation += 1
						self.Jobs_for_score[int(job[0])] = self.__wait_time[int(job[0])]

						if self.Jobs_Since_Last_Score_Computation == self.Score_Round_Frequency:

							best_score = float('inf')
							for predictor in self.Current_Predictors:
								score = predictor.get_score(self.Jobs_for_score)
								if best_score >= score:
									best_score = score
									self.Best_Predictor = predictor
							self.Jobs_for_score = {}
							self.Jobs_Since_Last_Score_Computation = 0
							#print 'Predictor score computation gives predictor', self.Best_Predictor.predictor_id
					 

			elif job[5] == "3" and int(job[2]) != -1:
				self.__running_job_id.remove(int(job[0]))
				self.__running_job_reqsize.remove(int(job[2]))
				self.__running_job_ert.remove(int(job[3]))	
				self.__running_job_cpu.remove(int(job[2]) * int(job[3]))				
				self.__free_nodes += int(job[2])
                                self.__elapsed_run_max = max(self.__elapsed_run_max, self.__real_run_time[int(job[0])])
                                self.__elapsed_run_min = min(self.__elapsed_run_min, self.__real_run_time[int(job[0])])

		
# Main Function
def main():
	start_time = time()
	if len(sys.argv) == 3:
		print "Starting...\n"
		system_state = Batch_System()
		system_state.Max_Nodes = int(sys.argv[2])
		system_state.convert_logs(sys.argv[1])
		system_state.log_parser()
	else:
		print "ERROR!! Insufficient Arguements... "
		print "python <python_file_name> <log_file_name> <max_nodes>"
	end_time = time()
	print "Done...\n\n"
	print "Running Time in seconds:" 
	print int(end_time-start_time)
	
if __name__ == '__main__':
	main()
