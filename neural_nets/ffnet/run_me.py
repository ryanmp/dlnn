'''
Information here...


'''

# global libraries
import scipy, numpy
from scipy import stats

# any global variables?
ncols = 73
nrows = 4100

# this is kind of a like a tolerance... training will continue until NF reaches this point
max_functions = 300

# anything smaller than nrows... and then save a few rows for a test (next line)
training_set_size = 3500

# we run the net on the next n rows after the training set 
in_sample_test_size = 50


# first and last must match in and out... but inner nodes can take any values
network_config = (ncols-11,ncols,ncols,ncols,1)


def main():
	
	import cPickle as pickle # persistant storage type

	print 'starting main'
	sorted_data = process_data()
	print 'saving data to pickle'
	pickle.dump( sorted_data, open( "data.p", "wb" ) )
	print 'loading data from pickle'
	sorted_data = pickle.load( open( "data.p", "rb" ) )
	
	forecast, real_data = build_ffnet(sorted_data)
	return forecast, real_data, sorted_data


def process_data():

	import datetime
	from dateutil import parser

	#datafile = open('SampleDataSmall.csv', 'r')
	datafile = open('contest_data1.out', 'r')
	data = []
	dts = []
	for idx, row in enumerate(datafile):
		if idx>1 and idx<nrows:
			if idx%500 == 0:
				print idx

			data_strings = row.strip().split('\t')
			#data_strings = row.strip().split(',')

			col_range = ncols # should be 72, but using a smaller set of columns for testing
			
			# place holder data
			data_floats = [0.0 for i in xrange(col_range)]
			
			for i in xrange(col_range): 
				# next I was planning on converting them all to floats,
				# but what should I do with the NULLs?
				try:
					tmp_float = float(data_strings[i])
				except:
					tmp_float = 0.0 # probably a bad idea to just use zero...
				
				data_floats[i] = tmp_float

			temp_dt = datetime.datetime(
					int(data_floats[3]),int(data_floats[4]),int(data_floats[5])
				)

			data_floats.append(temp_dt)

			data.append(data_floats)

			dts.append(temp_dt)

	data = numpy.array(data) 
	sorted_data = data[data[:,col_range].argsort()] # order by datetimes
	return sorted_data



def build_ffnet(sorted_data):

	from ffnet import ffnet, mlgraph
	from time import time
	from multiprocessing import cpu_count

	data_in_training2 = sorted_data[:training_set_size,10:-2].astype(float).tolist()
	data_target_training2 = [[i] for i in sorted_data[:training_set_size,0].astype(float)]

	print 'defining network'

	# Define net (large one)
	conec = mlgraph(network_config) #skipping first 11 cols
	net = ffnet(conec)

	#print "Sets initial weights via genetic algo"
	#net.train_genetic(data_in_training2, data_target_training2, individuals=3, generations=10)


	print "TRAINING NETWORK..."
	# that are many different training algos

	#net.train_rprop(data_in_training2, data_target_training2, a=1.2, b=0.5, mimin=1e-06, mimax=50.0, xmi=0.1, maxiter=40, disp=1)
	#net.train_momentum(data_in_training2, data_target_training2, eta=0.2, momentum=0.8, maxiter=200, disp=1)
	net.train_tnc(data_in_training2, data_target_training2, maxfun = max_functions, nproc = cpu_count(), messages=1)

	# doing an 'in sample' prediction from our trained nn
	data_input2 = sorted_data[training_set_size:training_set_size+in_sample_test_size,10:-2]
	nn_ans = net(data_input2)
	# shall we round it?
	nn_ans = [i[0] for i in nn_ans]
	#nn_ans = [round(i) for i in nn_ans]

	real_ans = sorted_data[training_set_size:training_set_size+in_sample_test_size,0].tolist()

	r_squared = scipy.stats.pearsonr(nn_ans, real_ans)[0]**2
	print r_squared

	return nn_ans, real_ans

	'''
	# do we want to graph it?

	import matplotlib.pyplot
	import seaborn

	fig = matplotlib.pyplot.figure()
	ax1 = fig.add_subplot(211)

	ax1.plot([i for i in xrange(len(nn_ans))],nn_ans, 'ro-', ms=.5)
	ax1.plot([i for i in xrange(len(nn_ans))],real_ans, 'go-', ms=.5)
	'''




if __name__ == '__main__':
	print 'starting main'
	forecast, real_data, sorted_data = main()










