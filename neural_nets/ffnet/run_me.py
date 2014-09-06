'''
Information here...


'''

# global libraries
import scipy, numpy, random
from scipy import stats

# any global variables?
ncols = 73
nrows = 1500

# this is kind of a like a tolerance... training will continue until NF reaches this point
max_functions = 2210

# anything smaller than nrows... and then save a few rows for a test (next line)
training_set_size = 1200

# we run the net on the next n rows after the training set 
in_sample_test_size = 100


# first and last must match in and out... but inner nodes can take any values
network_config = (ncols-11,12,1)


def main():
	
	import cPickle as pickle # persistant storage type

	print 'starting main'
	sorted_data = process_data()
	print 'saving data to pickle'
	pickle.dump( sorted_data, open( "data.p", "wb" ) )
	print 'loading data from pickle'
	sorted_data = pickle.load( open( "data.p", "rb" ) )
	
	net = build_ffnet(sorted_data)

	stats = []
	for i in xrange(10):
		tmp = calc_r2(net,sorted_data)
		stats.append(tmp[0])
		print tmp
	print sum(stats)/len(stats)
	
	show_comp(net,sorted_data)

	return net, sorted_data


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

	print "TRAINING NETWORK..."
	# that are many different training algos

	#net.train_rprop(data_in_training2, data_target_training2, a=1.2, b=0.5, mimin=1e-06, mimax=50.0, xmi=0.1, maxiter=40, disp=1)
	#net.train_momentum(data_in_training2, data_target_training2, eta=0.2, momentum=0.8, maxiter=200, disp=1)
	net.train_tnc(data_in_training2, data_target_training2, maxfun = max_functions, messages=1)
	#net.train_cg(data_in_training2, data_target_training2, maxiter=5, disp=1)
	#net.train_genetic(data_in_training2, data_target_training2, individuals=3, generations=300)
	#net.train_bfgs(data_in_training2, data_target_training2, maxfun = 86999, disp=1)

	return net




def calc_r2(net, sorted_data):

	start = random.randrange(0,len(sorted_data)-in_sample_test_size)
	end = start+in_sample_test_size

	real_ans = sorted_data[start:end,0].tolist()
	data_input2 = sorted_data[start:end,10:-2]
	
	nn_ans = [i[0] for i in net(data_input2)] # run the net
	
	#nn_ans = [round(i) for i in nn_ans] # shall we round it?

	r_squared = scipy.stats.pearsonr(nn_ans, real_ans)[0]**2
	return r_squared, start, end


def show_comp(net, sorted_data):

	start = 0
	end = len(sorted_data)-1

	real_ans = sorted_data[start:end,0].tolist()
	data_input2 = sorted_data[start:end,10:-2]
	
	nn_ans = [i[0] for i in net(data_input2)] # run the net
	
	import matplotlib.pyplot
	import seaborn

	fig = matplotlib.pyplot.figure()
	ax1 = fig.add_subplot(211)

	ax1.plot([i for i in xrange(len(nn_ans))], nn_ans, 'o-', linewidth=.3, markersize=3, color="red")
	ax1.plot([i for i in xrange(len(nn_ans))], real_ans, 'o-', linewidth=.3, markersize=3, color="blue")

	diffs = [j-i for (i,j) in zip(nn_ans,real_ans)]
	ax2 = fig.add_subplot(212)
	ax2.plot([i for i in xrange(len(diffs))], diffs, 'o-', linewidth=.3, markersize=3)

	fig.show()

	return real_ans, nn_ans



if __name__ == '__main__':
	print 'starting main'
	net, sorted_data = main()










