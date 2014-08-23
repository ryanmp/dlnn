

datafile = open('SampleDataSmall.csv', 'r')
data_in = []
data_target = []
for row in datafile:

	# logic for a single row
	data_strings = row.strip().split(',')
	data_floats = []

	for i in data_strings:
		# next I was planning on converting them all to floats,
		# but what should I do with the NULLs?
		try:
			tmp_float = float(i)
		except:
			tmp_float = 0.0 # probably a bad idea to just use zero...
		
		data_floats.append(tmp_float)

	data_in.append(data_floats[1:])
	data_target.append(data_floats[:1])

# first row had header data...
# just using the first 999 rows for training
data_in_training = data_in[1:1000]
data_target_training = data_target[1:1000]

'''
Okay, now that our data is prepared, let's run ffnet on it
'''

from ffnet import ffnet, mlgraph
from time import time
from multiprocessing import cpu_count


# Define net (large one)
conec = mlgraph((71,300,1))
net = ffnet(conec)

# Preserve original weights
weights0 = net.weights.copy()

print "TRAINING, this can take a while..."
for n in range(1, cpu_count()+1):
    net.weights[:] = weights0  #Start always from the same point
    t0 = time()
    net.train_tnc(data_in_training, data_target_training, maxfun = 50, nproc = n, messages=0)
    t1 = time()
    print '%s processes: %s s' %(n, t1-t0)


# I should dedicate a portion of the data to be untrained...
data_input2 = data_in[1000:]
ans = net(data_input2)
print ans





