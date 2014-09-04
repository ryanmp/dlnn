
import datetime
from dateutil import parser

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

import seaborn as sns #pretty graphs
sns.set(style="darkgrid")

import numpy # better arrays

#datafile = open('SampleDataSmall.csv', 'r')
datafile = open('contest_data1.out', 'r')
data = []
dts = []
for idx, row in enumerate(datafile):


	if idx>0:

		print idx

		# logic for a single row
		data_strings = row.strip().split('\t')
		#data_strings = row.strip().split(',')
		data_floats = [0.0 for i in xrange(72)]

		for i in xrange(72):
			# next I was planning on converting them all to floats,
			# but what should I do with the NULLs?
			try:
				tmp_float = float(data_strings[i])
			except:
				tmp_float = 0.0 # probably a bad idea to just use zero...
			
			if (i == 0): # fixing the infs that occasional appear in the s variable
				if float(data_strings[i]) > 1000:
					tmp_float = 0.0

			data_floats[i] = tmp_float

		data.append(data_floats)

		dts.append(datetime.datetime(
			int(data_floats[3]),int(data_floats[4]),int(data_floats[5])
			))

data = numpy.array(data) 

fig = plt.figure()

total = 30
ax1 = fig.add_subplot(total,1,1)
ax1.xaxis_date()
ax1.plot(dts, data[:,0], 'ro', ms=2.0) # just the first column

for i in xrange(total-1):

	print 'making plot: ', i

	temp_ax = fig.add_subplot(total,1,2+i)
	temp_ax.xaxis_date()
	temp_ax.plot(dts, data[:,10+i], 'o', ms=1.5)

	yticks = temp_ax.yaxis.get_major_ticks()
	yticks[0].label1.set_visible(False)
	yticks[-1].label1.set_visible(False)

fig.subplots_adjust(left=0.05, right=.99, top=0.99, bottom=0.00)
fig.subplots_adjust(hspace=.02,wspace=0) 
fig.autofmt_xdate()
#fig.show()
fig.set_size_inches(10,30)
fig.savefig('test.png',dpi=100)

