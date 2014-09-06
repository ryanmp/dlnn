import datetime
from dateutil import parser

import numpy # better arrays
import cPickle as pickle # persistant storage type

#datafile = open('SampleDataSmall.csv', 'r')
datafile = open('contest_data1.out', 'r')
data = []
dts = []
for idx, row in enumerate(datafile):
	if idx>200000:# and idx<10000:
		print idx

		data_strings = row.strip().split('\t')
		#data_strings = row.strip().split(',')

		col_range = 10 # should be 72, but using a smaller set of columns for testing
		
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

print 'saving data to pickle'
#pickle.dump( sorted_data, open( "data.p", "wb" ) )

#print 'loading data from pickle'
#sorted_data = pickle.load( open( "data.p", "rb" ) )


