'''
Author: Ryan Phillips
Date: 9-6-2014


'''

# global libraries
import scipy, numpy, random, math
from scipy import stats
import logging
import matplotlib.pyplot
import seaborn

# there are 4 different retail ids, 1-4
# and 3 different item ids, 1-3

# any global variables?

#retailer 1-4
retailer = 1
#item 1-3
item = 1

ncols = 72
nrows = 4000 # 287420 is the first 9999999

col_training_set = [3] + [i for i in xrange(10,72)] # cols w r^2 > .05
# this can also be used to just grab a set of columns, like =[3, 10, 30, ...]

l = len(col_training_set)

# this is kind of a like a tolerance... training will continue until NF reaches this point
max_functions = 400

# used for the genetic training algo
max_population = 17

# anything smaller than nrows...
# the rest wil be used for out_of_sample_testing
training_set_size = 3000

# first and last must match in and out... but inner nodes can take any values
#network_config = (ncols-11,ncols-11,15,1)

#network_config = (l,128,64,32,16,8,4,2,1)
network_config = (l,l,l,2,1)


def main():

	logging.basicConfig(filename='run_me.log', level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',)

	import cPickle as pickle # persistant storage type

	'''
	sorted_data = process_data()
	print 'saving data to pickle'
	pickle.dump( sorted_data, open( "data.p", "wb" ) )
	print 'loading data from pickle'
	'''
	sorted_data = pickle.load( open( "data.p", "rb" ) )
	print 'loaded ',len(sorted_data),' observations'


	#cols = find_corr(sorted_data)
	#print cols
	#x = [i[0] for i in cols]

	# overwriting globals with best set of cols
	#col_training_set = x[1:30]
	#l = len(col_training_set)
	#network_config = (l,8,4,2,1)

	graph_data(sorted_data, [0])

	logging.info('starting with params... max_functions: ' + str(max_functions) + ', total_rows: ' + str(nrows) + ', training_set_size: ' + str(training_set_size))
	logging.info('col set:' + str(col_training_set))

	net = build_ffnet(sorted_data)
	calc_stats(net,sorted_data)
	graph_it(net,sorted_data)

	return net, sorted_data


def find_corr(data):
	results = []
	for i in xrange(ncols):
		print 'calc CORR for: ',i
		r_sqr = scipy.stats.pearsonr(data[:,i], data[:,0])[0]**2
		if math.isnan(r_sqr):
			r_sqr = 0.0
		results.append((i,r_sqr))
	results = sorted(results,key=lambda x: x[1], reverse=True)
	return results


def graph_data(data, which_cols):

	fig = matplotlib.pyplot.figure()
	total = len(which_cols)
	for i in xrange(total):

		print 'making plot: ', i
		temp_ax = fig.add_subplot(total,2,i*2-1)
		#temp_ax.xaxis_date()
		temp_ax.plot([j for j in xrange(len(data))], data[:,which_cols[i]], 'o', ms=1.5)

		#temp_ax.plot([i for i in xrange(len(data))], [i for i in xrange(len(data))], 'o', ms=1.5)

		yticks = temp_ax.yaxis.get_major_ticks()
		yticks[0].label1.set_visible(False)
		yticks[-1].label1.set_visible(False)

		ax3 = fig.add_subplot(total,2,i*2 + 2)
		ax3.scatter(data[:,which_cols[i]], data[:,0], s=1)
		ax3.set_title('col: ' + str(which_cols[i]))

	'''
	fig.subplots_adjust(left=0.05, right=.99, top=0.99, bottom=0.00)
	fig.subplots_adjust(hspace=.02,wspace=0)
	'''
	fig.autofmt_xdate()

	fig.tight_layout()
	fig.show()


def process_data():

	import datetime
	from dateutil import parser

	#datafile = open('SampleDataSmall.csv', 'r')
	datafile = open('contest_data1.out', 'r')
	data = []
	dts = []
	for idx, row in enumerate(datafile):
		if idx>1:
			if idx%1000 == 0:
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


			# just running 2,2 for now
			if (data_floats[0] < 1000):
				if (data_floats[1] == 2.0):
					if (data_floats[2] == 1.0):
						data_floats.append(temp_dt)
						data.append(data_floats)
						dts.append(temp_dt)


	print 'TOTAL DATA SET SIZE:', len(data)

	data = numpy.array(data)
	sorted_data = data[data[:,col_range].argsort()] # order by datetimes

	print 'reducing it to:', len(sorted_data[:nrows])

	return sorted_data[:nrows]



def build_ffnet(sorted_data):

	from ffnet import ffnet, imlgraph, mlgraph
	from time import time
	from multiprocessing import cpu_count
	import networkx
	import pylab

	data_in_training2 = sorted_data[:training_set_size,10:-2].astype(float).tolist()
	data_target_training2 = [[i] for i in sorted_data[:training_set_size,0].astype(float)]

	new_data_in = sorted_data[:training_set_size,col_training_set[0]]
	for i in col_training_set[1:]:
		new_data_in = numpy.column_stack((new_data_in, sorted_data[:training_set_size,i]))
	data_in_training2 = new_data_in.astype(float).tolist()

	print 'defining network'

	# Define net (large one)
	conec = mlgraph(network_config, biases=False) #skipping first 11 cols

	net = ffnet(conec)

	#print 'draw network'
	#networkx.draw_graphviz(net.graph, prog='dot')
	#pylab.show()

	logging.info('network built as: ' + str(network_config) )

	print "TRAINING NETWORK..."
	# that are many different training algos

	#net.train_rprop(data_in_training2, data_target_training2, a=1.9, b=0.1, mimin=1e-06, mimax=15.0, xmi=0.5, maxiter=max_functions, disp=1)
	###net.train_momentum(data_in_training2, data_target_training2, eta=0.2, momentum=0.1, maxiter=max_functions, disp=1)
	net.train_tnc(data_in_training2, data_target_training2, maxfun = max_functions, messages=1)
	#net.train_cg(data_in_training2, data_target_training2, maxiter=max_functions, disp=1)
	#net.train_genetic(data_in_training2, data_target_training2, individuals=max_population, generations=max_functions)
	#net.train_bfgs(data_in_training2, data_target_training2, maxfun = max_functions, disp=1)

	return net


def calc_stats(net, sorted_data):

	print 'calc stats'

	start = 0
	end = len(sorted_data)-1

	real_ans = sorted_data[start:end,0].tolist()

	new_data_in = sorted_data[:,col_training_set[0]]
	for i in col_training_set[1:]:
		new_data_in = numpy.column_stack((new_data_in, sorted_data[:,i]))
	data_in_training2 = new_data_in.astype(float).tolist()
	data_input2 = data_in_training2[start:end]
	#data_input2 = sorted_data[start:end,10:-2]

	nn_ans = [i[0] for i in net(data_input2)]

	r_squared_in = scipy.stats.pearsonr(nn_ans[:training_set_size], real_ans[:training_set_size])[0]**2
	r_squared_out = scipy.stats.pearsonr(nn_ans[training_set_size:], real_ans[training_set_size:])[0]**2

	print('in: ' + str(r_squared_in) + ', out:' + str(r_squared_out))
	logging.info('in: ' + str(r_squared_in) + ', out:' + str(r_squared_out))


def graph_it(net, sorted_data):

	print 'graph it'

	start = 0
	end = len(sorted_data)-1

	real_ans = sorted_data[start:end,0].tolist()


	new_data_in = sorted_data[:,col_training_set[0]]
	for i in col_training_set[1:]:
		new_data_in = numpy.column_stack((new_data_in, sorted_data[:,i]))
	data_in_training2 = new_data_in.astype(float).tolist()
	data_input2 = data_in_training2[start:end]
	#data_input2 = sorted_data[start:end,10:-2]

	nn_ans = [i[0] for i in net(data_input2)] # run the net

	fig = matplotlib.pyplot.figure()

	ax1 = fig.add_subplot(221)
	ax1.plot([i for i in xrange(len(nn_ans))], nn_ans, 'o-', linewidth=.3, markersize=3, color="red")
	ax1.plot([i for i in xrange(len(nn_ans))], real_ans, 'o-', linewidth=.3, markersize=3, color="blue")
	ax1.set_ylabel('s')
	ax1.set_title('comparing real_s and nn_out_s')

	diffs = [j-i for (i,j) in zip(nn_ans,real_ans)]
	ax2 = fig.add_subplot(223)
	ax2.plot([i for i in xrange(len(diffs[:training_set_size]))], diffs[:training_set_size], 'o-', linewidth=.3, markersize=3)
	ax2.plot([i+training_set_size for i in xrange(len(diffs[training_set_size:]))], diffs[training_set_size:], 'o-', linewidth=.3, markersize=3)
	ax2.set_ylabel('s')
	ax2.set_title('real_s - nn_out_s')

	ax3 = fig.add_subplot(222)
	ax3.scatter(real_ans[:training_set_size], nn_ans[:training_set_size], s=1)
	ax3.set_xlabel('real')
	ax3.set_ylabel('nn_out')
	ax3.set_title('in sample')

	ax4 = fig.add_subplot(224)
	ax4.scatter(real_ans[training_set_size:], nn_ans[training_set_size:], s=1)
	ax4.set_xlabel('real')
	ax4.set_ylabel('nn_out')
	ax4.set_title('out of sample')

	fig.tight_layout()
	fig.show()


if __name__ == '__main__':
	print 'starting main'
	net, sorted_data = main()
