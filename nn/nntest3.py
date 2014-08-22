from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet,UnsupervisedDataSet
from pybrain.structure import LinearLayer
ds = SupervisedDataSet(21, 21)

#data = [1,2,4,6,2,3,4,5,1 3 5 6 7 1 4 7 1 2 3 5 6,1 2 5 6 2 4 4 5 1 2 5 6 7 1 4 6 1 2 3 3 6]

ds.addSample(map(int,'1 2 4 6 2 3 4 5 1 3 5 6 7 1 4 7 1 2 3 5 6'.split()),map(int,'1 2 5 6 2 4 4 5 1 2 5 6 7 1 4 6 1 2 3 3 6'.split()))
ds.addSample(map(int,'1 2 5 6 2 4 4 5 1 2 5 6 7 1 4 6 1 2 3 3 6'.split()),map(int,'1 3 5 7 2 4 6 7 1 3 5 6 7 1 4 6 1 2 2 3 7'.split()))
net = buildNetwork(21, 20, 21, outclass=LinearLayer,bias=True, recurrent=True)
trainer = BackpropTrainer(net, ds)
trainer.trainEpochs(10)


x = [ i for i in net.activateOnDataset(ds)[0]]
print x

import matplotlib.pyplot
import seaborn

fig = matplotlib.pyplot.figure()
ax1 = fig.add_subplot(211)
ax1.plot(x, [i for i in xrange(len(x))], 'k', label='Actual')