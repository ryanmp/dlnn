import numpy # better arrays
import cPickle as pickle # persistant storage type

import neurolab

print 'loading data from pickle'
#data = pickle.load( open( "data.p", "rb" ) )

idx = numpy.where(data==999999.0)[0][0] # the first 99999999
# happens to equal 287420

'''
input_data = []
target_data = []
for i in xrange(0,10):
	input_data.append( data[:5+i,3:-2] )
	target_data.append( data[5:10+i,1:2] )
	'''



input_data =  data[:5,3:-2].astype(float)
target_data = data[5:10,2].astype(float)

size = len(input_data)

input1 = np.array([input_data]).reshape(5 * 1, 1)
target1 = np.array([target_data]).reshape(5 * 1, 1)

net = neurolab.net.newff([[-7000, 7000]],[5, 1])

# Train network
error = net.train(input1, target1, epochs=500, show=100)
### ARGH! I can't get this stupid thing to train properly


#in_sample_test = []
#for i in xrange(50,60):
#	in_sample_test.append( data[:5+i,3:-2] )


in_sample_test = data[20:25,3:-2].astype(float)

# Simulate network
out = net.sim(in_sample_test)
