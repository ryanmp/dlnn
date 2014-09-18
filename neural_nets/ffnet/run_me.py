'''
Author: Ryan Phillips
Date: 9-6-2014


'''

# global libraries
import scipy, numpy, random, math, os
from scipy import stats
import logging
import matplotlib.pyplot
import seaborn

# there are 4 different retail ids, 1-4
# and 3 different item ids, 1-3

# any global variables?


# for running it just on specific retailer/item
retailer = 2 #retailer 1-4
item = 2 #item 1-3

# just describes source data
ncols = 72

# total for this test...
nrows = 49030 # 287420 is the first 9999999

col_training_set = [i for i in xrange(1,72)] + [73]  # cols w r^2 > .05
# this can also be used to just grab a set of columns, like =[3, 10, 30, ...]

l = len(col_training_set)

# these two determine the maximum training time
max_loops = 7
min_loops = 1
max_total = 400
max_functions = 3 # per loop

# used for the genetic training algo
max_population = 17

# anything smaller than nrows...
# the rest wil be used for out_of_sample_testing
training_set_size = 43000

overfitting_threshold = .3

# first and last must match in and out... but inner nodes can take any values
#network_config = (ncols-11,ncols-11,15,1)
#network_config = (l,128,64,32,16,8,4,2,1)

network_config = (l,l*3,1)

def main():

	logging.basicConfig(filename='run_me.log', level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',)

	import cPickle as pickle # persistant storage type

	sorted_data_full = process_data()
	#print 'saving data to pickle'
	#pickle.dump( sorted_data_full, open( "data.p", "wb" ) )
	#print 'loading data from pickle'
	#sorted_data_full = pickle.load( open( "data.p", "rb" ) )
	#print 'loaded ',len(sorted_data_full),' observations'

	sorted_data = []
	for i in sorted_data_full:
		if i[1] == retailer and i[2] == item:
			sorted_data.append(i)
	print 'total with this selection:', len(sorted_data)
	sorted_data = sorted_data[:nrows]
	sorted_data = numpy.array(sorted_data)

	print 'just running on first: ',len(sorted_data),' observations'

	#cols = find_corr(sorted_data)
	#print cols
	#x = [i[0] for i in cols]

	# overwriting globals with best set of cols
	#col_training_set = x[1:30]
	#l = len(col_training_set)
	#network_config = (l,8,4,2,1)

	#graph_data(sorted_data, [0])

	logging.info('starting with params... max_functions: ' + str(max_functions) + ', total_rows: ' + str(nrows) + ', training_set_size: ' + str(training_set_size))
	logging.info('running on retailer/item: ' + str(retailer) + ' ' +  str(item))
	logging.info('col set:' + str(col_training_set))

	#net = None
	net = build_ffnet(sorted_data)
	#calc_stats(net,sorted_data)
	graph_it(net,sorted_data)
	graph_weekly(net,sorted_data)

	return net, sorted_data, sorted_data_full


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

			temp_week = temp_dt.isocalendar()[0] * 52
			temp_week += temp_dt.isocalendar()[1]

			# just running 2,2 for now
			if (data_floats[0] < 1000):
				data_floats.append(temp_dt)
				data_floats.append(temp_week)

				data.append(data_floats)


	print 'TOTAL DATA SET SIZE:', len(data)

	data = numpy.array(data)
	sorted_data = data[data[:,col_range].argsort()] # order by datetimes

	return sorted_data



def build_ffnet(sorted_data):

	logging.info('starting new run! -----------------------------')
	print 'defining network'

	from ffnet import ffnet, imlgraph, mlgraph, loadnet, savenet
	from time import time
	from multiprocessing import cpu_count
	import networkx
	import pylab

	#data_in_training2 = sorted_data[:training_set_size,10:-2].astype(float).tolist()
	data_target_training2 = [[i] for i in sorted_data[:training_set_size,0].astype(float)]

	new_data_in = sorted_data[:training_set_size,col_training_set[0]]
	for i in col_training_set[1:]:
		new_data_in = numpy.column_stack((new_data_in, sorted_data[:training_set_size,i]))
	data_in_training2 = new_data_in.astype(float).tolist()

	# Define net (large one)
	conec = mlgraph(network_config, biases=False) #skipping first 11 cols
	net = ffnet(conec)
	print 'saving initialized net'
	savenet(net, 'starting_net.n')
	#net = loadnet('starting_net.n') # this way we can init a complex net just once

	#print 'draw network'
	#networkx.draw_graphviz(net.graph, prog='dot')
	#pylab.show()

	logging.info('network built as: ' + str(network_config) )

	print "TRAINING NETWORK..."
	# that are many different training algos

	#net.train_rprop(data_in_training2, data_target_training2, a=1.9, b=0.1, mimin=1e-06, mimax=15.0, xmi=0.5, maxiter=max_functions, disp=1)
	###net.train_momentum(data_in_training2, data_target_training2, eta=0.2, momentum=0.1, maxiter=max_functions, disp=1)
	stats = []
	biggest = 0
	total = 0
	try:
		for i in xrange(min_loops,max_loops):
			total += max_functions+i
			if total>max_total:
				break
			print 'training for:',max_functions+i, "total is:", total

			net.train_tnc(data_in_training2, data_target_training2, maxfun = max_functions+i, messages=1)
			#net.train_rprop(data_in_training2, data_target_training2, a=1.2, b=0.5, mimin=1e-06, mimax=50.0, xmi=0.1, maxiter=max_functions*20, disp=1)

			in0, out0, s1, s2 = calc_stats(net,sorted_data)
			stats.append((in0, out0,total, s1, s2))
			#if out0<=(biggest/1.4) and in0>.7:
			if out0<=(biggest/9) and in0>overfitting_threshold:
				print 'breaking out early'
				break
			if out0 > biggest: # found a new best
				biggest = out0
				savenet(net, 'best_net.n')
	except KeyboardInterrupt: # this way command-c just breaks out of this loop
		pass


	#net.train_cg(data_in_training2, data_target_training2, maxiter=max_functions, disp=1)
	#net.train_genetic(data_in_training2, data_target_training2, individuals=max_population, generations=max_functions)
	#net.train_bfgs(data_in_training2, data_target_training2, maxfun = max_functions, disp=1)
	stats = sorted(stats, reverse=True, key=lambda x: x[1])
	for i in stats:
		print i[1], i[2], i[3], i[4]

	net = loadnet('best_net.n')
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

	s1 = sum(nn_ans[:training_set_size])
	s2 = sum(real_ans[:training_set_size])
	print 'i.s. sums (nn/real):',s1,s2

	s1 = sum(nn_ans[training_set_size:])
	s2 = sum(real_ans[training_set_size:])
	print 'o.s. sums (nn/real):',s1,s2

	r_squared_in = scipy.stats.pearsonr(nn_ans[:training_set_size], real_ans[:training_set_size])[0]**2
	r_squared_out = scipy.stats.pearsonr(nn_ans[training_set_size:], real_ans[training_set_size:])[0]**2
	print('r^2 in: ' + str(r_squared_in) + ', out:' + str(r_squared_out))
	logging.info('r^2 in: ' + str(r_squared_in) + ', out:' + str(r_squared_out))

	wmape_in = wmape(nn_ans[:training_set_size], real_ans[:training_set_size])
	wmape_out = wmape(nn_ans[training_set_size:], real_ans[training_set_size:])
	print('mape in: ' + str(wmape_in) + ', out:' + str(wmape_out))
	logging.info('mape in: ' + str(wmape_in) + ', out:' + str(wmape_out))

	start_dt = min(sorted_data[:,-2])
	end_dt = max(sorted_data[:,-2])
	middle_dt = sorted_data[training_set_size:training_set_size+1,-2][0]

	start_week = start_dt.isocalendar()[0] * 52
	start_week += start_dt.isocalendar()[1]
	end_week = end_dt.isocalendar()[0] * 52
	end_week += end_dt.isocalendar()[1]
	middle_week = middle_dt.isocalendar()[0] * 52
	middle_week += middle_dt.isocalendar()[1]
	week_range = end_week - start_week + 1
	middle_week_split = week_range - middle_week

	weekly_real_s = [0 for i in xrange(week_range)]
	weekly_nn_s = [0 for i in xrange(week_range)]
	for i,j in zip(sorted_data,nn_ans):
		temp_idx = i[-1] - start_week
		weekly_real_s[temp_idx] += i[0]
		weekly_nn_s[temp_idx] += j

	r_squared_weekly_all = scipy.stats.pearsonr(weekly_real_s, weekly_nn_s)[0]**2
	r_squared_weekly_in = scipy.stats.pearsonr(weekly_real_s[:middle_week_split], weekly_nn_s[:middle_week_split])[0]**2
	r_squared_weekly_out = scipy.stats.pearsonr(weekly_real_s[middle_week_split:], weekly_nn_s[middle_week_split:])[0]**2


	print 'r^2 weekly_all/in/out: ', r_squared_weekly_all,r_squared_weekly_in,r_squared_weekly_out
	logging.info('r^2 weekly_all/in/out: '+ str(r_squared_weekly_all) + str(r_squared_weekly_in) + str(r_squared_weekly_out))

	mape_weekly_all = wmape(weekly_real_s, weekly_nn_s)
	print 'mape weekly_all: ', mape_weekly_all
	logging.info('mape weekly_all: '+ str(mape_weekly_all))

	return r_squared_in, r_squared_out, s1, s2


def wmape(xs,ys):
	total = 0
	count = 0
	for f,a in zip(xs,ys):
		if a>0: # don't know what to do with zeroes... so I'm just excluding them
			total += abs((a-f)/a)
			count += 1

	ret = -999999
	if count>0:
		ret = float(total)/count

	return ret


def graph_weekly(net, sorted_data):

	start = 0
	end = len(sorted_data)-1

	new_data_in = sorted_data[:,col_training_set[0]]
	for i in col_training_set[1:]:
		new_data_in = numpy.column_stack((new_data_in, sorted_data[:,i]))
	data_in_training2 = new_data_in.astype(float).tolist()

	data_input2 = data_in_training2[start:end]
	#data_input2 = sorted_data[start:end,10:-2]

	nn_ans = [i[0] for i in net(data_input2)]

	start_dt = min(sorted_data[:,-2])
	end_dt = max(sorted_data[:,-2])
	middle_dt = sorted_data[training_set_size:training_set_size+1,-2][0]

	start_week = start_dt.isocalendar()[0] * 52
	start_week += start_dt.isocalendar()[1]
	end_week = end_dt.isocalendar()[0] * 52
	end_week += end_dt.isocalendar()[1]
	middle_week = middle_dt.isocalendar()[0] * 52
	middle_week += middle_dt.isocalendar()[1]
	week_range = end_week - start_week + 1
	middle_week_split = middle_week - start_week

	weekly_real_s = [0 for i in xrange(week_range)]
	weekly_nn_s = [0 for i in xrange(week_range)]
	for i,j in zip(sorted_data,nn_ans):
		temp_idx = i[-1] - start_week
		weekly_real_s[temp_idx] += i[0]
		weekly_nn_s[temp_idx] += j

	fig = matplotlib.pyplot.figure()

	ax1 = fig.add_subplot(221)
	ax1.plot([i for i in xrange(len(weekly_nn_s))], weekly_nn_s, 'o-', linewidth=.3, markersize=3, color="red")
	ax1.plot([i for i in xrange(len(weekly_nn_s))], weekly_real_s, 'o-', linewidth=.3, markersize=3, color="blue")
	ax1.set_ylabel('s')
	ax1.set_title('comparing real_s (blue) and nn_out_s (red)')

	diffs = [j-i for (i,j) in zip(weekly_real_s,weekly_nn_s)]
	ax2 = fig.add_subplot(223)
	ax2.plot([i for i in xrange(len(diffs[:middle_week_split]))], diffs[:middle_week_split], 'o-', linewidth=.3, markersize=3)
	ax2.plot([i+middle_week_split for i in xrange(len(diffs[middle_week_split:]))], diffs[middle_week_split:], 'o-', linewidth=.3, markersize=3)
	ax2.set_ylabel('s')
	ax2.set_title('real_s - nn_out_s')

	ax3 = fig.add_subplot(222)
	ax3.scatter(weekly_real_s[:middle_week_split], weekly_nn_s[:middle_week_split], s=1)
	ax3.set_xlabel('real')
	ax3.set_ylabel('nn_out')
	ax3.set_title('in sample')

	ax4 = fig.add_subplot(224)
	ax4.scatter(weekly_real_s[middle_week_split:], weekly_nn_s[middle_week_split:], s=1)
	ax4.set_xlabel('real')
	ax4.set_ylabel('nn_out')
	ax4.set_title('out of sample')

	fig.tight_layout()
	os.system('say "check out that graph"')
	fig.show()


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
	os.system('say "check out that graph"')
	fig.show()

if __name__ == '__main__':
	print 'starting main'
	net, sorted_data, sorted_data_full = main()


